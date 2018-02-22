# this demos the various compositing algorithms found in cvtools

import cvtools
import matplotlib.pyplot as pl
import numpy as np
from PIL import Image as image
import skimage.transform

path = 'im/compositing/'

# run the multi-resolution blending demo
im1 = cvtools.imread(path+'apple.png')
im2 = cvtools.imread(path+'orange.png')
[ny,nx,_] = im1.shape
mask = np.zeros((ny,nx))
mask[:,:int(nx/2)] = 1
m = cvtools.Multi_Res_Blending(im1, im2, mask)
pl.figure(1, (8,2))
pl.subplot(1,4,1)
pl.imshow(im1)
pl.axis('off')
pl.title('source', fontsize=8, weight='demibold')
pl.subplot(1,4,2)
pl.imshow(im2)
pl.axis('off')
pl.title('target', fontsize=8, weight='demibold')
pl.subplot(1,4,3)
pl.imshow(mask,cmap='gray')
pl.axis('off')
pl.title('mask', fontsize=8, weight='demibold')
pl.subplot(1,4,4)
pl.imshow(m.I)
pl.axis('off')
pl.title('blended image', fontsize=8, weight='demibold')
pl.show()

# run graph-cut compositing demo
im1 = cvtools.imread(path+'strawberries-left.png')
im2 = cvtools.imread(path+'strawberries-right.png')

# for test purposes
ny = 128
nx = int(ny*im1.shape[1]/im1.shape[0])
im1 = skimage.transform.resize(im1,(ny,nx),mode='constant')
im2 = skimage.transform.resize(im2,(ny,nx),mode='constant')

[ny,nx,_] = im1.shape # im1 and im2 are the same shape
mask1, mask2 = np.zeros((ny,nx)), np.zeros((ny,nx))
mask1[1:ny-1,1:48] = 1
mask2[1:ny-1,-48:-1] = 1

m = cvtools.Graph_Cut_Compositing([im1,im2], [mask1,mask2])
pl.imshow(m.mcomb)
pl.show()

