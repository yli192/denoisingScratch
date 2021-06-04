#! /home/local/PARTNERS/yl715/anaconda3/bin/python
import matplotlib.pyplot as plt
import numpy as np
from sys import argv
#displays 3D image of .img format
from skimage.util import montage as montage2d


if len(argv) <= 1:
    print("arguments: *.img")
    exit()
filename = argv[1]
fp = open(filename,'rb')
img= np.load(fp)
img_zFirst = np.swapaxes(img,0,2)
print("image shape (z,y,x):",img.shape)
#fig,axes = plt.subplots(nrows=1,ncols=3)
#slice_indices = [11,31,211]

# for ind in range(3):
#     slice = img[slice_indices[ind],:,:]
#     axes[ind].imshow(slice,cmap='gray')
#     axes[ind].axis('off')


# fig, ax2 = plt.subplots(1,1, figsize = (40, 40))
# ax2.imshow(montage2d(img))
fig, ax1 = plt.subplots(1,1, figsize = (40, 40))
ax1.imshow(img[300,:,:])
plt.show()
