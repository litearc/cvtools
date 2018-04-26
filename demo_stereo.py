# this demos concepts related to epipolar geometry

import cv
from cv.util import colors, tcolors
import matplotlib.pyplot as pl
import numpy as np
import scipy as sp
import sys
from PIL import Image as im

path = 'im/epipolar/'

# run epipolar geometry demo

# load data
m1 = cv.util.imread(path+'1.jpg')
p1 = cv.util.get_labeled_pts(path+'1-points.png').T # [{x,y} points]
m2 = cv.util.imread(path+'2.jpg')
p2 = cv.util.get_labeled_pts(path+'2-points.png').T # [{x,y} points]

# get epipolar lines ..........................................................
F = cv.stereo.get_fundamental_matrix(p1, p2)
s1,b1,s2,b2 = cv.stereo.get_epipolar_lines(p1, p2, F)
e1,e2 = cv.stereo.get_epipoles(F)

[ny,nx,nc] = m1.shape
xl, xu = 0, nx-1
yl1, yu1 = s1*xl+b1, s1*xu+b1
yl2, yu2 = s2*xl+b2, s2*xu+b2

# draw figures
pl.figure(1,(16,8))
pl.subplot(1,2,1)
pl.imshow(m1)
pl.plot([xl,xu], [yl1,yu1], color = tcolors['blue'])
pl.plot(p1[0,:], p1[1,:], color=colors['red'], marker='.', ls='None')
pl.axis('off')
pl.title('image 1', fontsize=8, weight='demibold')
pl.subplot(1,2,2)
pl.imshow(m2)
pl.plot([xl,xu], [yl2,yu2], color = tcolors['blue'])
pl.plot(p2[0,:], p2[1,:], color=colors['red'], marker='.', ls='None')
pl.axis('off')
pl.title('image 2', fontsize=8, weight='demibold')
pl.show()

# rectify images ..............................................................
m1rect, m2rect, H1, H2 = cv.stereo.rectify_images(m1, p1, m2, p2)

# find points and epipolar lines in rectified images
p1 = cv.stereo.tform_pts(p1, H1)
p2 = cv.stereo.tform_pts(p2, H2)

n = yl1.size
pl1 = np.array(cv.stereo.tform_pts(np.vstack((xl*np.ones(n), yl1)), H1))
pu1 = np.array(cv.stereo.tform_pts(np.vstack((xu*np.ones(n), yu1)), H1))
pl2 = np.array(cv.stereo.tform_pts(np.vstack((xl*np.ones(n), yl2)), H2))
pu2 = np.array(cv.stereo.tform_pts(np.vstack((xu*np.ones(n), yu2)), H2))

# show rectified images
pl.figure(1,(16,8))
pl.subplot(1,2,1)
pl.imshow(m1rect)
pl.plot([pl1[0,:],pu1[0,:]], [pl1[1,:],pu1[1,:]], color = tcolors['blue'])
pl.plot(p1[0,:], p1[1,:], color=colors['red'], marker='.', ls='None')
pl.axis('off')
pl.xlim([0, nx-1])
pl.ylim([ny-1, 0])
pl.title('image 1 rectified', fontsize=8, weight='demibold')
pl.subplot(1,2,2)
pl.imshow(m2rect)
pl.plot([pl2[0,:],pu2[0,:]], [pl2[1,:],pu2[1,:]], color = tcolors['blue'])
pl.plot(p2[0,:], p2[1,:], color=colors['red'], marker='.', ls='None')
pl.axis('off')
pl.xlim([0, nx-1])
pl.ylim([ny-1, 0])
pl.title('image 2 rectified', fontsize=8, weight='demibold')
pl.show()

