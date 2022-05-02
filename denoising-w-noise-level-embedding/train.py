"""
## Author: Gary. Y Li
## A Noise-level-aware Framework for PET Image Denoising
## Ye Li, Jianan Cui, Junyu Chen, Guodong Zeng, Scott Wollenweber, Floris Jansen, Se-In Jang, Kyungsang Kim, Kuang Gong, Quanzheng Li
## https://arxiv.org/abs/2203.08034
"""
import os
from config import Config
opt = Config('training3D_MixedLoss_1_8_full_p32_SUV_SubDS.yml')
gpus = ','.join([str(i) for i in opt.GPU])
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpus

import torch
torch.backends.cudnn.benchmark = True

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import random
import time
import numpy as np

import utils
from data import get_training_data3d_wNoiseInfo_l1, get_validation_data3d_wNoiseInfo_l1

from MPRNet3D_wNLE import ORSNet3D_wNLEEncDec_ORSOnly_OneMLP
import losses
from warmup_scheduler import GradualWarmupScheduler


######### Set Seeds ###########
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)

start_epoch = 1
mode = opt.MODEL.MODE
session = opt.MODEL.SESSION

result_dir = os.path.join(opt.TRAINING.SAVE_DIR, mode, 'results', session)
model_dir  = os.path.join(opt.TRAINING.SAVE_DIR, mode, 'models',  session)

utils.mkdir(result_dir)
utils.mkdir(model_dir)

train_dir = opt.TRAINING.TRAIN_DIR
val_dir   = opt.TRAINING.VAL_DIR

######### Model ###########
model_restoration = ORSNet3D_wNLEEncDec_ORSOnly_OneMLP(in_c=1,in_NLF=1,n_feat=96)
pytorch_total_params = sum(p.numel() for p in model_restoration.parameters())
print(pytorch_total_params)
model_restoration.cuda()

device_ids = [i for i in range(torch.cuda.device_count())]
if torch.cuda.device_count() > 1:
  print("\n\nLet's use", torch.cuda.device_count(), "GPUs!\n\n")


new_lr = opt.OPTIM.LR_INITIAL

optimizer = optim.Adam(model_restoration.parameters(), lr=new_lr, betas=(0.9, 0.999),eps=1e-8, weight_decay=1e-8)

######### Scheduler ###########
warmup_epochs = 3
scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.OPTIM.NUM_EPOCHS-warmup_epochs+40, eta_min=opt.OPTIM.LR_MIN)
scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
scheduler.step()

######### Resume ###########
if opt.TRAINING.RESUME:
    path_chk_rest    = utils.get_last_path(model_dir, 'model_epoch_5.pth')
    utils.load_checkpoint(model_restoration,path_chk_rest)
    start_epoch = utils.load_start_epoch(path_chk_rest) + 1
    utils.load_optim(optimizer, path_chk_rest)

    for i in range(1, start_epoch):
        scheduler.step()
    new_lr = scheduler.get_lr()[0]
    print('------------------------------------------------------------------------------')
    print("==> Resuming Training with learning rate:", new_lr)
    print('------------------------------------------------------------------------------')

if len(device_ids)>1:
    model_restoration = nn.DataParallel(model_restoration, device_ids = device_ids)

######### Loss ###########

criterion_char = losses.CharbonnierLoss()
criterion_edge = losses.EdgeLoss()
######### DataLoaders ###########
train_dataset = get_training_data3d_wNoiseInfo_l1(train_dir, {'patch_size':opt.TRAINING.TRAIN_PS})
train_loader = DataLoader(dataset=train_dataset, batch_size=opt.OPTIM.BATCH_SIZE, shuffle=True, num_workers=16, drop_last=False, pin_memory=True)

val_dataset = get_validation_data3d_wNoiseInfo_l1(val_dir, {'patch_size':opt.TRAINING.VAL_PS})
val_loader = DataLoader(dataset=val_dataset, batch_size=opt.OPTIM.BATCH_SIZE, shuffle=False, num_workers=8, drop_last=False, pin_memory=True)

print('===> Start Epoch {} End Epoch {}'.format(start_epoch,opt.OPTIM.NUM_EPOCHS + 1))
print('===> Loading datasets')

best_psnr = 0
best_epoch = 0
best_iter = 0

eval_now = len(train_loader)//3 - 1
epoch_evl_interval = 5
print(f"\nEval after every {eval_now} Iterations !!!\n")
mixup = utils.MixUp_AUG3D()

for epoch in range(start_epoch, opt.OPTIM.NUM_EPOCHS + 1):
    epoch_start_time = time.time()
    epoch_loss = 0
    train_id = 1
    print('epoch',epoch)
    model_restoration.train()
    for i, data in enumerate(train_loader, 0):

        for param in model_restoration.parameters():
            param.grad = None

        noiseLevel_input = torch.stack(data[4],dim=1).float().cuda()

        # 1/8 to 1/4 input and output
        target_intermediate = torch.unsqueeze(data[0].cuda(),1).float() # 1/4 image
        input_ = torch.unsqueeze(data[3].cuda(),1).float() #1/8 image

        # 1/4 to Full input and output
        target = torch.unsqueeze(data[1].cuda(), 1).float()  # full noise
        input_intermediate = torch.unsqueeze(data[2].cuda(), 1).float()  # 1/4 image


        if epoch> 10:
            target, input_ = mixup.aug(target, input_)

        noiseLevel_input_LC_low, noiseLevel_input_LC_mid = torch.split(noiseLevel_input, [1, 1], dim=1)

        restored = model_restoration(input_,noiseLevel_input_LC_low)
        loss_list_edge = [criterion_edge(torch.clamp(restored[j], -100, 100), target) for j in range(len(restored))]
        loss_list_char = [criterion_char(torch.clamp(restored[j], -100, 100), target) for j in range(len(restored))]

        total_loss_edge = 0.0
        total_loss_char = 0.0

        for loss in loss_list_edge:
            total_loss_edge += loss
        for loss in loss_list_char:
            total_loss_char += loss
        total_loss = total_loss_char + 0.05 * total_loss_edge

        # Compute loss at each stage
        total_loss.backward()
        optimizer.step()
        epoch_loss += total_loss.item()

        ### Evaluation ####
        if i%eval_now==0 and i>0 and (epoch in [1,6,25] or epoch%epoch_evl_interval==0):
            print('evaluation begins')
            model_restoration.eval()
            psnr_val_rgb = []
            psnr_val_rgb_input = []
            for ii, data_val in enumerate((val_loader), 0):

                target_intermediate = torch.unsqueeze(data_val[0].cuda(),1).float() # 1/4 dose image
                target = torch.unsqueeze(data_val[1].cuda(),1).float()  # full dose image
                input_intermediate = torch.unsqueeze(data_val[2].cuda(),1).float() #1/4 dose image
                input_ = torch.unsqueeze(data_val[3].cuda(),1).float() #1/8 dose image
                noiseLevel_input = torch.stack(data_val[4], dim=1).float().cuda()

                with torch.no_grad():
                    noiseLevel_input_LC_low, noiseLevel_input_LC_mid = torch.split(noiseLevel_input, [1, 1], dim=1)


                    restored = model_restoration(input_, noiseLevel_input_LC_low)
                    psnr_val_rgb.append(utils.torchPSNR_3D(target, restored[0]))
                    psnr_val_rgb_input.append(utils.torchPSNR_3D(target, input_))


            psnr_val_rgb  = torch.stack(psnr_val_rgb).mean().item()
            psnr_val_rgb_input  = torch.stack(psnr_val_rgb_input).mean().item()


            if psnr_val_rgb > best_psnr:
                best_psnr = psnr_val_rgb
                best_epoch = epoch
                best_iter = i
                torch.save({'epoch': epoch,
                            'state_dict': model_restoration.state_dict(),
                            'optimizer' : optimizer.state_dict()
                            }, os.path.join(model_dir,"model_best.pth"))

            print("[epoch %d it %d restored_PSNR: %.4f ]" % (epoch, i, psnr_val_rgb))
            print("[epoch %d it %d input_PSNR: %.4f]" % (epoch, i, psnr_val_rgb_input))


            model_restoration.train()

    scheduler.step()

    print("------------------------------------------------------------------")
    print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(epoch, time.time()-epoch_start_time, epoch_loss, scheduler.get_lr()[0]))
    print("------------------------------------------------------------------")

    torch.save({'epoch': epoch,
                'state_dict': model_restoration.state_dict(),
                'optimizer' : optimizer.state_dict()
                }, os.path.join(model_dir,"model_latest.pth"))

