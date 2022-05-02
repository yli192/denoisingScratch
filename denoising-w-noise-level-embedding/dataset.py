import os
import numpy as np
from torch.utils.data import Dataset
import torch
from PIL import Image
import torchvision.transforms.functional as TF
from pdb import set_trace as stx
import random

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['jpeg', 'npy','JPEG', 'jpg', 'png', 'JPG', 'PNG', 'gif'])

class RandomCrop3D():
    def __init__(self, img_sz, crop_sz):
        h, w, d = img_sz
        assert (h, w, d) > crop_sz
        self.img_sz = tuple((h, w, d))
        self.crop_sz = tuple(crop_sz)

    def __call__(self, x):
        slice_hwd = [self._get_slice(i, k) for i, k in zip(self.img_sz, self.crop_sz)]
        return self._crop(x, *slice_hwd)

    @staticmethod
    def _get_slice(sz, crop_sz):
        try:
            lower_bound = torch.randint(sz - crop_sz, (1,)).item()
            return lower_bound, lower_bound + crop_sz
        except:
            return (None, None)

    @staticmethod
    def _crop(x, slice_h, slice_w, slice_d):
        return x[slice_h[0]:slice_h[1], slice_w[0]:slice_w[1], slice_d[0]:slice_d[1]]

class DataLoaderTrain3d_wNoiseInfo_l1(Dataset):
    def __init__(self, rgb_dir, img_options=None):
        super(DataLoaderTrain3d_wNoiseInfo_l1, self).__init__()

        inp_files = sorted(os.listdir(os.path.join(rgb_dir)))
        # tar_files = sorted(os.listdir(os.path.join(rgb_dir, 'target')))

        self.inp_filenames = [os.path.join(rgb_dir, x) for x in inp_files if is_image_file(x)]
        # self.tar_filenames = [os.path.join(rgb_dir, 'target', x) for x in tar_files if is_image_file(x)]

        self.img_options = img_options
        self.sizex = len(self.inp_filenames)  # get the size of target

        self.ps = self.img_options['patch_size']

        patch_counts_low_all = []
        for file in self.inp_filenames:
            patch_counts_low = os.path.basename(file).split('_')[2]
            patch_counts_low_all.append(int(patch_counts_low))

        patch_counts_lows = np.asarray(patch_counts_low_all)
        self.patch_counts_low_max = np.max(patch_counts_lows)
        self.patch_counts_low_min = np.min(patch_counts_lows)

        patch_counts_mid_all = []
        for file in self.inp_filenames:
            patch_counts_mid = os.path.basename(file).split('_')[3]
            patch_counts_mid_all.append(int(patch_counts_mid))
        patch_counts_mids = np.asarray(patch_counts_mid_all)
        self.patch_counts_mid_max = np.max(patch_counts_mids)
        self.patch_counts_mid_min = np.min(patch_counts_mids)
        self.bins_mid = [0, 1000, 2000, 15000, 25000, 40000, self.patch_counts_mid_max]
        self.bins_low = [0, 1000, 2000, 15000, 25000, 40000, self.patch_counts_low_max]
        self.noise_histogram_low = np.zeros(6, )
        self.noise_histogram_mid = np.zeros(6, )

    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        index_ = index % self.sizex
        ps = self.ps

        #noiseLevel_dict = self.noiseLevel_dict

        inp_path = self.inp_filenames[index_]
        #case_num = os.path.basename(inp_path).strip().split('_')[0]

        patch_count_low = os.path.basename(inp_path).strip().split('_')[2]

        patch_count_mid = os.path.basename(inp_path).strip().split('_')[3]

        noiseLevelInfo = np.zeros(2,)

        for n in range(0, len(self.bins_mid) - 1):
            b_start = self.bins_mid[n]
            b_end = self.bins_mid[n + 1]
            if int(patch_count_mid) <= b_end and int(patch_count_mid) > b_start:
                #print(n, patch_count_mid, b_start, b_end)
                noiseLevelInfo[0] = n
                self.noise_histogram_low[n] += 1

        for n in range(0, len(self.bins_low) - 1):
            b_start = self.bins_low[n]
            b_end = self.bins_low[n + 1]
            if int(patch_count_low) <= b_end and int(patch_count_low) > b_start:
                #print(n, patch_count_low, b_start, b_end)
                noiseLevelInfo[1] = n
                self.noise_histogram_mid[n] += 1

        noiseLevelInfo_local = [round(num, 1) for num in noiseLevelInfo]

        # tar_path = self.tar_filenames[index_]

        inp_img_all = np.load(inp_path)
        inp_img = inp_img_all[0, :, :, :]
        #ct_img = inp_img_all[3, :, :, :]
        inp_img_intermediate = inp_img_all[1, :, :, :]
        tar_img_intermediate = inp_img_all[1, :, :, :]
        tar_img =  inp_img_all[2, :, :, :]
        # print(inp_img.shape)
        # a = np.asarray(inp_img)
        w, h, d = tar_img.shape
        padw = ps - w if w < ps else 0
        padh = ps - h if h < ps else 0
        padd = ps - d if d < ps else 0

        # Reflect Pad in case image is smaller than patch_size
        if padw != 0 or padh != 0 or padd != 0:
            inp_img = TF.pad(inp_img, (0, 0, padw, padh, padd), padding_mode='reflect')
            #ct_img = TF.pad(ct_img, (0, 0, padw, padh, padd), padding_mode='reflect')
            inp_img_intermediate = TF.pad(inp_img_intermediate, (0, 0, padw, padh, padd), padding_mode='reflect')
            tar_img = TF.pad(tar_img, (0, 0, padw, padh, padd), padding_mode='reflect')
            tar_img_intermediate = TF.pad(tar_img_intermediate, (0, 0, padw, padh, padd), padding_mode='reflect')

        inp_img = TF.to_tensor(
            inp_img)  # 3,256,256 to_tensor converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] if the PIL Image belongs to one of the modes (L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1) or if the numpy.ndarray has dtype = np.uint8
        inp_img_intermediate = TF.to_tensor(inp_img_intermediate)
        tar_img = TF.to_tensor(tar_img)
        tar_img_intermediate = TF.to_tensor(tar_img_intermediate)
        #ct_img = TF.to_tensor(ct_img)

        hh, ww, dd = tar_img.shape[0], tar_img.shape[1], tar_img.shape[2]

        rr = random.randint(0, hh - ps)
        cc = random.randint(0, ww - ps)
        tt = random.randint(0, dd - ps)
        aug = random.randint(0, 8)

        # Crop patch
        inp_img = inp_img[rr:rr + ps, cc:cc + ps, tt:tt + ps]
        inp_img_intermediate = inp_img_intermediate[rr:rr + ps, cc:cc + ps, tt:tt + ps]

        tar_img = tar_img[rr:rr + ps, cc:cc + ps, tt:tt + ps]
        tar_img_intermediate = tar_img_intermediate[rr:rr + ps, cc:cc + ps, tt:tt + ps]
        #ct_img = ct_img[rr:rr + ps, cc:cc + ps, tt:tt + ps]


        # Data Augmentations
        if aug == 1:
            inp_img = inp_img.flip(1)
            inp_img_intermediate = inp_img_intermediate.flip(1)
            tar_img = tar_img.flip(1)
            tar_img_intermediate = tar_img_intermediate.flip(1)
            #ct_img = ct_img.flip(1)

        elif aug == 2:
            inp_img = inp_img.flip(2)
            #ct_img = ct_img.flip(2)
            inp_img_intermediate = inp_img_intermediate.flip(2)

            tar_img = tar_img.flip(2)
            tar_img_intermediate = tar_img_intermediate.flip(2)
        elif aug == 3:
            inp_img = torch.rot90(inp_img, dims=(0, 1))
            #ct_img = torch.rot90(ct_img, dims=(0, 1))

            inp_img_intermediate = torch.rot90(inp_img_intermediate, dims=(0, 1))

            tar_img = torch.rot90(tar_img, dims=(0, 1))
            tar_img_intermediate = torch.rot90(tar_img_intermediate, dims=(0, 1))
        elif aug == 4:
            inp_img = torch.rot90(inp_img, dims=(0, 1), k=3)
            inp_img_intermediate = torch.rot90(inp_img_intermediate, dims=(0, 1), k=3)

            tar_img = torch.rot90(tar_img, dims=(0, 1), k=3)
            tar_img_intermediate = torch.rot90(tar_img_intermediate, dims=(0, 1), k=3)
        elif aug == 5:
            inp_img = torch.rot90(inp_img, dims=(0, 1), k=3)
            #ct_img = torch.rot90(ct_img, dims=(0, 1), k=3)

            inp_img_intermediate = torch.rot90(inp_img_intermediate, dims=(0, 1), k=3)

            tar_img = torch.rot90(tar_img, dims=(0, 1), k=3)
            tar_img_intermediate = torch.rot90(tar_img_intermediate, dims=(0, 1), k=3)
        elif aug == 6:
            inp_img = torch.rot90(inp_img.flip(1), dims=(0, 1))
            #ct_img = torch.rot90(ct_img.flip(1), dims=(0, 1))

            inp_img_intermediate = torch.rot90(inp_img_intermediate.flip(1), dims=(0, 1))

            tar_img = torch.rot90(tar_img.flip(1), dims=(0, 1))
            tar_img_intermediate = torch.rot90(tar_img_intermediate.flip(1), dims=(0, 1))
        elif aug == 7:
            inp_img = torch.rot90(inp_img.flip(2), dims=(0, 1))
            #ct_img = torch.rot90(ct_img.flip(2), dims=(0, 1))

            inp_img_intermediate = torch.rot90(inp_img_intermediate.flip(2), dims=(0, 1))

            tar_img = torch.rot90(tar_img.flip(2), dims=(0, 1))
            tar_img_intermediate = torch.rot90(tar_img_intermediate.flip(2), dims=(0, 1))

        filename = os.path.splitext(os.path.split(inp_path)[-1])[0]

        return tar_img_intermediate, tar_img, inp_img_intermediate, inp_img, noiseLevelInfo_local, filename



class DataLoaderVal3d_wNoiseInfo_l1(Dataset):
    def __init__(self, rgb_dir, img_options=None, rgb_dir2=None):
        super(DataLoaderVal3d_wNoiseInfo_l1, self).__init__()

        inp_files = sorted(os.listdir(os.path.join(rgb_dir)))
        #random.shuffle(inp_files)
        inp_files = random.sample(set(inp_files), 1000)
        #tar_files = sorted(os.listdir(os.path.join(rgb_dir, 'target')))

        self.inp_filenames = [os.path.join(rgb_dir,  x)  for x in inp_files if is_image_file(x)]
        #self.tar_filenames = [os.path.join(rgb_dir, 'target', x) for x in tar_files if is_image_file(x)]

        self.img_options = img_options
        self.sizex       = len(self.inp_filenames)  # get the size of target

        self.ps = self.img_options['patch_size']

        patch_counts_low_all = []
        for file in self.inp_filenames:
            patch_counts_low = os.path.basename(file).split('_')[2]
            patch_counts_low_all.append(int(patch_counts_low))

        patch_counts_lows = np.asarray(patch_counts_low_all)
        self.patch_counts_low_max = np.max(patch_counts_lows)
        self.patch_counts_low_min = np.min(patch_counts_lows)

        patch_counts_mid_all = []
        for file in self.inp_filenames:
            patch_counts_mid = os.path.basename(file).split('_')[3]
            patch_counts_mid_all.append(int(patch_counts_mid))
        patch_counts_mids = np.asarray(patch_counts_mid_all)
        self.patch_counts_mid_max = np.max(patch_counts_mids)
        self.patch_counts_mid_min = np.min(patch_counts_mids)
        self.bins_mid = [0, 1000, 2000, 15000, 25000, 40000, self.patch_counts_mid_max]
        self.bins_low = [0, 1000, 2000, 15000, 25000, 40000, self.patch_counts_low_max]
        self.noise_histogram_low = np.zeros(6, )
        self.noise_histogram_mid = np.zeros(6, )

    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        index_ = index % self.sizex
        ps = self.ps

        inp_path = self.inp_filenames[index_]
        #tar_path = self.tar_filenames[index_]
        patch_count_low = os.path.basename(inp_path).strip().split('_')[2]

        patch_count_mid = os.path.basename(inp_path).strip().split('_')[3]

        noiseLevelInfo = np.zeros(2, )

        for n in range(0, len(self.bins_mid) - 1):
            b_start = self.bins_mid[n]
            b_end = self.bins_mid[n + 1]
            if int(patch_count_mid) <= b_end and int(patch_count_mid) > b_start:
                #print(n, patch_count_mid, b_start, b_end)
                noiseLevelInfo[0] = n
                self.noise_histogram_low[n] += 1

        for n in range(0, len(self.bins_low) - 1):
            b_start = self.bins_low[n]
            b_end = self.bins_low[n + 1]
            if int(patch_count_low) <= b_end and int(patch_count_low) > b_start:
                #print(n, patch_count_low, b_start, b_end)
                noiseLevelInfo[1] = n
                self.noise_histogram_mid[n] += 1

        noiseLevelInfo_local = [round(num, 1) for num in noiseLevelInfo]

        inp_img_all = np.load(inp_path)
        inp_img = inp_img_all[0, :, :, :]
        #ct_img = inp_img_all[3, :, :, :]

        inp_img_intermediate= inp_img_all[1, :, :, :]
        # print(inp_img_all.shape)
        tar_img_intermediate = inp_img_all[1, :, :, :]
        tar_img = inp_img_all[2, :, :, :]

        # inp_img = inp_img_all[0, :, :, :]
        # # tar_img = Image.open(tar_path)
        # tar_img = inp_img_all[2, :, :, :]
        inp_img = TF.to_tensor(inp_img)
        tar_img = TF.to_tensor(tar_img)
        inp_img_intermediate = TF.to_tensor(inp_img_intermediate)
        tar_img_intermediate = TF.to_tensor(tar_img_intermediate)

        if self.ps is not None:
            rand_crop = RandomCrop3D(inp_img_intermediate.shape, (ps, ps, ps))

            inp_img = rand_crop(inp_img)
            tar_img = rand_crop(tar_img)
            inp_img_intermediate = rand_crop(inp_img_intermediate)
            tar_img_intermediate = rand_crop(tar_img_intermediate)

        filename = os.path.splitext(os.path.split(inp_path)[-1])[0]

        return tar_img_intermediate,tar_img, inp_img_intermediate, inp_img, noiseLevelInfo_local, filename



class DataLoaderTest3d(Dataset):
    def __init__(self, inp_dir, img_options):
        super(DataLoaderTest3d, self).__init__()

        inp_files = sorted(os.listdir(inp_dir))
        self.inp_filenames = [os.path.join(inp_dir, x) for x in inp_files if is_image_file(x)]

        self.inp_size = len(self.inp_filenames)
        self.img_options = img_options

    def __len__(self):
        return self.inp_size

    def __getitem__(self, index):

        path_inp = self.inp_filenames[index]
        filename = os.path.splitext(os.path.split(path_inp)[-1])[0]
        inp_img_all = np.load(filename)
        inp_img = inp_img_all[0, :, :, :]


        inp = TF.to_tensor(inp_img)
        return inp, filename

