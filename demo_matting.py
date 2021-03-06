# this demos the various matting algorithms found in cvtools

import cv
import matplotlib.pyplot as pl
import numpy as np
import skimage.transform

# load data
path = 'im/matting/'
img = cv.util.imread(path+'inp.png') # raw image
gta = cv.util.imread(path+'gt.png')  # ground-truth alpha
tm = cv.util.imread(path+'tm.png')   # trimap

# process trimap
tm[(tm!=0)&(tm!=1)] = .5 # all 'unknown' pixels have value = .5
if tm.ndim == 2: # convert grayscale to rgb
  tm = np.transpose(np.tile(tm,(3,1,1)),((1,2,0)))

# option to scale down image to reduce computation time
fs = .25 # scale factor
if fs != 1.0:
  [nx,ny,_] = img.shape
  sz = (int(nx*fs),int(ny*fs),3) # new size
  img = skimage.transform.resize(img,sz,mode='constant')
  gta = skimage.transform.resize(gta,sz,mode='constant')
  tm = skimage.transform.resize(tm,sz,mode='constant')
  tm[(tm!=0)&(tm!=1)] = .5

# run bayesian matting demo
a = cv.matting.bayesian(img, tm.copy())
pl.figure(1, (6,2))
pl.subplot(1,4,1)
pl.imshow(img)
pl.axis('off')
pl.title('image', fontsize=8, weight='demibold')
pl.subplot(1,4,2)
pl.imshow(a*img)
pl.axis('off')
pl.title('foreground', fontsize=8, weight='demibold')
pl.subplot(1,4,3)
pl.imshow(gta)
pl.axis('off')
pl.title('ground truth', fontsize=8, weight='demibold')
pl.subplot(1,4,4)
pl.imshow(a)
pl.axis('off')
pl.title('calculated alpha', fontsize=8, weight='demibold')

# run closed-form matting demo
a = cv.matting.natural(img, tm.copy())
pl.figure(2, (6,2))
pl.subplot(1,4,1)
pl.imshow(img)
pl.axis('off')
pl.title('image', fontsize=8, weight='demibold')
pl.subplot(1,4,2)
pl.imshow(a*img)
pl.axis('off')
pl.title('foreground', fontsize=8, weight='demibold')
pl.subplot(1,4,3)
pl.imshow(gta)
pl.axis('off')
pl.title('ground truth', fontsize=8, weight='demibold')
pl.subplot(1,4,4)
pl.imshow(a)
pl.axis('off')
pl.title('calculated alpha', fontsize=8, weight='demibold')

pl.show()

