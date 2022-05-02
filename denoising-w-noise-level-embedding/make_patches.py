"""
## Author: Gary. Y Li
## Description: This script makes patches from an image with a given overlapping step; The input directories contains the normalized(to the SUV) PET images.
## Last Modified: Feb 23, 2022
"""

import numpy as np
import os
import nibabel as nib
from patchify import patchify
from skimage import filters
from skimage.metrics import structural_similarity as ssim
counter = 0
highPet_train_path = '/home/local/PARTNERS/yl715/data/garyli/fullpet_jqtest_352_SUV/'
lowPet_train_path = '/home/local/PARTNERS/yl715/data/garyli/1_8pet_jqtest_352_SUV/'
intermediatePet_train_path = '/home/local/PARTNERS/yl715/data/garyli/1_4pet_jqtest_352_SUV/'
ct_train_path = '/home/local/PARTNERS/yl715/data/garyli/ct_jqtest_352_SUV/'
output_patch_path = '/home/local/PARTNERS/yl715/data/garyli/p32x32x32_patches_1_8_1_4_full_ct_test_SUV_CountsConcentrationinMask/'

for img in os.listdir(ct_train_path):
    if img.endswith('.nii.gz'):
        case_num = img.strip().split('_')[1].split('.')[0]
        ct_img = nib.load(ct_train_path+'/'+img)
        ct_img = ct_img.get_fdata()

        noise_img = nib.load(lowPet_train_path + '/' + img)
        noise_img = noise_img.get_fdata()

        highPet_img = nib.load(highPet_train_path + '/' + img)
        highPet_img = highPet_img.get_fdata()

        intermediatePet_img = nib.load(intermediatePet_train_path + '/' + img)
        intermediatePet_img = intermediatePet_img.get_fdata()

        vertical_patches_ct_img = patchify(ct_img, (32, 208, 128), step=32) #split the whole image vertically
        vertical_patches_noise_img = patchify(noise_img, (32, 208, 128), step=32)
        vertical_patches_intermediatePet_img = patchify(intermediatePet_img, (32, 208, 128), step=32)
        vertical_patches_highPet_img = patchify(highPet_img, (32, 208, 128), step=32)


        for m in range(vertical_patches_ct_img.shape[0]):
            lbm_surrogate = int(np.sum(np.sum(np.sum(vertical_patches_ct_img[m, :, :, :, :, :]))))

            _ = vertical_patches_ct_img[m, 0, 0, :, :, :]
            patches_ct_img = patchify(_, (32, 32, 32), step=16)

            _ = vertical_patches_noise_img[m, 0, 0, :, :, :]
            patches_noise_img = patchify(_, (32, 32, 32), step=16)

            _ = vertical_patches_intermediatePet_img[m, 0, 0, :, :, :]
            patches_intermediatePet_img = patchify(_, (32, 32, 32), step=16)

            _ = vertical_patches_highPet_img[m, 0, 0, :, :, :]
            patches_highPet_img = patchify(_, (32, 32, 32), step=16)


            for s in range(patches_noise_img.shape[0]):
                for j in range(patches_noise_img.shape[1]):
                    for k in range(patches_noise_img.shape[2]):
                        ct = patches_ct_img[s, j, k, :, :, :]
                        GT = patches_highPet_img[s, j, k, :, :, :]
                        lowPET = patches_noise_img[s, j, k, :, :, :]
                        threshold_lowPET = filters.threshold_otsu(lowPET)
                        binary_mask = lowPET > threshold_lowPET
                        masked_lowPET = lowPET * binary_mask
                        averageCount_in_patch_low = int(np.sum(np.sum(np.sum(masked_lowPET)))/np.sum(np.sum(np.sum(binary_mask))))
                        averageCount_count_mask = binary_mask * averageCount_in_patch_low
                        SSIM = round(np.abs(ssim(lowPET, averageCount_count_mask)),2)
                        intermediatePet = patches_intermediatePet_img[s, j, k, :, :, :]
                        counter = counter + 1
                        stacked_img = np.stack([lowPET,intermediatePet,GT,ct],axis=0)
                        np.save(output_patch_path  + case_num + '_'+str(averageCount_in_patch_low) +'_'+ str(SSIM)+'_' + str(counter) +'.npy', stacked_img)

        print(case_num)
