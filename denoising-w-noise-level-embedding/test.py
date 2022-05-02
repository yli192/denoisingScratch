"""
## Author: Gary. Y Li
## A Noise-level-aware Framework for PET Image Denoising
## Ye Li, Jianan Cui, Junyu Chen, Guodong Zeng, Scott Wollenweber, Floris Jansen, Se-In Jang, Kyungsang Kim, Kuang Gong, Quanzheng Li
## https://arxiv.org/abs/2203.08034
"""
import numpy as np
import os
import argparse
import skimage.metrics
import torch
import torch.nn as nn
import utils
from patchify import patchify, unpatchify
from MPRNet3D_wNLE import ORSNet3D_wNLEEncDec_ORSOnly_OneMLP


parser = argparse.ArgumentParser(description='PET Image Denoising using MPRNet w/ Noise Level Emberring')

parser.add_argument('--GTs_dir', default='/home/local/PARTNERS/yl715/data/garyli/jq_test_1_8_1_4_full_ct_SUV/', type=str, help='Directory of GT images')

parser.add_argument('--result_dir', default='/home/local/PARTNERS/yl715/prelim_HO_data/3DORSOnly_MSEEDGEModulatedbypcInfo_PS32_nfeat96_1_8_to_Full_wpcInfo_ModEncDecORS_PET_OneMLP_FullDS_l1_SUV_v3/Denoising/results/MPRNet/', type=str, help='Directory for results')
parser.add_argument('--weights', default='/home/local/PARTNERS/yl715/prelim_HO_data/3DORSOnly_MSEEDGEModulatedbypcInfo_PS32_nfeat96_1_8_to_Full_wpcInfo_ModEncDecORS_PET_OneMLP_FullDS_l1_SUV_v3/Denoising/models/MPRNet/model_best.pth', type=str, help='Path to weights')
parser.add_argument('--gpus', default='0', type=str, help='CUDA_VISIBLE_DEVICES')
parser.add_argument('--save_images', action='store_true', help='Save denoised images in result directory')
quater_flag = False

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

result_dir = os.path.join(args.result_dir)
utils.mkdir(result_dir)

if args.save_images:
    result_dir_img = os.path.join(args.result_dir, 'images')
    utils.mkdir(result_dir_img)

model_restoration = MPRNet3D_wNLEEncDec_ORSOnly_OneMLP(in_c=1,in_NLF=1,n_feat=96)

utils.load_checkpoint(model_restoration,args.weights)
print("===>Testing using weights: ",args.weights)
model_restoration.cuda()
model_restoration = nn.DataParallel(model_restoration)
model_restoration.eval()

#####patch validation#####
filepath_GTs = os.path.join(args.GTs_dir)
GTs = []
for _ in os.listdir(filepath_GTs):
    GTs.append(_)

if quater_flag:
    filepath_val = os.path.join(args.input_dir)
    all = []
    for _ in os.listdir(filepath_val):
        all.append(_)
else:
    all = GTs


val = all
PSNR = []
PSNR_LD = []
SSIM = []
SSIM_LD =[]

patch_counts_low_all = []
for file in os.listdir('/home/local/PARTNERS/yl715/data/garyli/p32x32x32_patches_1_8_1_4_full_ct_SUV'):
    patch_counts_low = os.path.basename(file).split('_')[2]
    patch_counts_low_all.append(int(patch_counts_low))

patch_counts_lows = np.asarray(patch_counts_low_all)
patch_counts_low_max = np.max(patch_counts_lows)
patch_counts_low_min = np.min(patch_counts_lows)

patch_counts_mid_all = []
for file in os.listdir('/home/local/PARTNERS/yl715/data/garyli/p32x32x32_patches_1_8_1_4_full_ct_SUV'):
    patch_counts_mid = os.path.basename(file).split('_')[3]
    patch_counts_mid_all.append(int(patch_counts_mid))
patch_counts_mids = np.asarray(patch_counts_mid_all)
patch_counts_mid_max = np.max(patch_counts_mids)
patch_counts_mid_min = np.min(patch_counts_mids)
bins_mid = [0, 1000, 2000, 15000, 25000, 40000, patch_counts_mid_max]
bins_low = [0, 1000, 2000, 15000, 25000, 40000, patch_counts_low_max]
noise_histogram_low = np.zeros(6, )
noise_histogram_mid = np.zeros(6, )

with torch.no_grad():
    patch_size = 32
    overlap_size = 8
    text_file = os.path.join(result_dir, "Val_results_p64_o32_1_8_to_Full.txt")
    with open(text_file, "w") as outfile:
        outfile.write('ImgName PSNR_input PSNR_restored SSIM_input SSIM_restored \n')

        for j in val:

            img_name = os.path.splitext(j)[0]
            case_num = img_name.split('_')[1]

            noisy_img = np.load(filepath_GTs + j)[0, :, :, :]
            highPet_img = np.load(filepath_GTs + j)[2, :, :, :]
            ct_img = np.load(filepath_GTs + j)[3, :, :, :]

            vertical_patches_ct_img = patchify(ct_img, (32, 224, 128), step=32)  # split the whole image vertically
            vertical_patches_noise_img = patchify(noisy_img, (32, 224, 128), step=32)
            vertical_patches_patches_highPet_img = patchify(highPet_img, (32, 224, 128), step=32)
            restored_whole_image = np.zeros_like(vertical_patches_noise_img)

            for m in range(vertical_patches_ct_img.shape[0]):
                ##calculate lbm_surrogate for each row of the image

                _ = vertical_patches_ct_img[m, 0, 0, :, :, :]
                patches_ct_img = patchify(_, (patch_size, patch_size, patch_size), step=overlap_size)

                _ = vertical_patches_noise_img[m, 0, 0, :, :, :]
                patches_noise_img = patchify(_, (patch_size, patch_size, patch_size), step=overlap_size)

                _ = vertical_patches_patches_highPet_img[m, 0, 0, :, :, :]
                patches_highPet_img = patchify(_, (patch_size, patch_size, patch_size), step=overlap_size)

                restored_row_patches = np.zeros_like(patches_noise_img)
                counter = 0
                for s in range(patches_noise_img.shape[0]):
                    for l in range(patches_noise_img.shape[1]):
                        for k in range(patches_noise_img.shape[2]):
                            patch_count_low = int(np.sum(np.sum(np.sum(patches_noise_img[s, l, k, :, :, :]))))

                            noiseLevelInfo = np.zeros(1, )

                            for n in range(0, len(bins_low) - 1):
                                b_start = bins_low[n]
                                b_end = bins_low[n + 1]
                                if int(patch_count_low) <= b_end and int(patch_count_low) > b_start:
                                    noiseLevelInfo[0] = n
                                    print(patch_count_low,n,b_start,b_end)


                            noiseLevel_input = torch.Tensor(noiseLevelInfo).float().cuda()

                            noiseLevel_in_out_final = torch.unsqueeze(noiseLevel_input, 0)
                            print(noiseLevel_in_out_final)

                            noisy_patch_cuda = torch.from_numpy(patches_noise_img[s, l, k, :, :, :]).unsqueeze(
                                0).unsqueeze(0).cuda().float()
                            GT_patch_cuda = torch.from_numpy(patches_highPet_img[s, l, k, :, :, :]).unsqueeze(0).unsqueeze(
                                0).cuda().float()
                            restored_patch = model_restoration(noisy_patch_cuda, noiseLevel_in_out_final)
                            restored_img = restored_patch[0]
                            #restored_img = noisy_patch_cuda

                            print('PSNR of restroed patch {}:{}'.format(img_name,
                                                                        utils.torchPSNR_3D(GT_patch_cuda,
                                                                                           restored_img)))
                            print('PSNR of LD patch {}:{}'.format(img_name,
                                                                  utils.torchPSNR_3D(GT_patch_cuda, noisy_patch_cuda)))

                            counter += 1
                            restored_row_patches[s, l, k, :, :, :] = torch.clamp(restored_img, -100, 100).cpu().detach().squeeze(
                                0).squeeze(0)

                row_restored_image = unpatchify(restored_row_patches, _.shape)

                restored_whole_image[m, 0, 0, :, :, :] = row_restored_image

            whole_restored_image = unpatchify(restored_whole_image, noisy_img.shape)

            PSRN_noisy_img= skimage.metrics.peak_signal_noise_ratio(highPet_img,noisy_img,data_range=np.max(np.max(np.max(highPet_img))))
            SSIM_noisy_img = skimage.metrics.structural_similarity(highPet_img,noisy_img)
            PSNR_LD.append(PSRN_noisy_img)
            SSIM_LD.append(SSIM_noisy_img)

            PSNR_restored_img = skimage.metrics.peak_signal_noise_ratio(highPet_img, whole_restored_image, data_range=np.max(np.max(np.max(highPet_img))))
            SSIM_restored_img = skimage.metrics.structural_similarity(highPet_img, whole_restored_image)

            outfile.write('{} {:.2f} {:.2f} {:.2f} {:.2f} \n'.format(img_name,PSRN_noisy_img,PSNR_restored_img,SSIM_noisy_img,SSIM_restored_img))
            print('PSNR and SSIM beafter denoising {}:{}'
                  .format(img_name,skimage.metrics.peak_signal_noise_ratio(highPet_img, whole_restored_image, data_range=np.max(np.max(np.max(highPet_img))))))
            print('SSIM after denoising {}:{}'.format(img_name, skimage.metrics.structural_similarity(highPet_img, whole_restored_image)))
            PSNR.append(PSNR_restored_img)
            SSIM.append(SSIM_restored_img)

            if args.save_images:
                save_file = os.path.join(result_dir_img, '{}_restored_1_8_to_full.npy'.format(img_name))
                np.save(save_file, whole_restored_image)
                save_file = os.path.join(result_dir_img, '{}_GT_full.npy'.format(img_name))
                np.save(save_file, np.load(filepath_GTs + j)[2, :, :, :])
                save_file = os.path.join(result_dir_img, '{}_input1_8.npy'.format(img_name))
                np.save(save_file, np.load(filepath_GTs + j)[0, :, :, :])

        PSNR = np.asarray(PSNR)
        PSNR_LD = np.asarray(PSNR_LD)
        SSIM = np.asarray(SSIM)
        SSIM_LD = np.asarray(SSIM_LD)

        outfile.write('mean PSNR of validation data {}\n'.format(np.mean(PSNR)))
        outfile.write('mean PSNR_LD of validation data {}\n'.format(np.mean(PSNR_LD)))

        outfile.write('mean SSIM of validation data {}\n'.format(np.mean(SSIM)))
        outfile.write('mean SSIM_LD of validation data {}\n'.format(np.mean(SSIM_LD)))

    outfile.close()