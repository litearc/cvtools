# this demos concepts related to epipolar geometry

import cv
import matplotlib.pyplot as pl
import numpy as np

path = 'im/epipolar/'

# run epipolar geometry demo
im1 = cv.util.imread(path+'1.jpg')
pt1 = cv.util.get_labeled_pts(path+'1-points.png')
im2 = cv.util.imread(path+'2.jpg')
pt2 = cv.util.get_labeled_pts(path+'2-points.png')

F = cv.stereo.get_fundamental_matrix(pt1, pt2)
m1,b1,m2,b2 = cv.stereo.get_epipolar_lines(pt1, pt2, F)
e1,e2 = cv.stereo.get_epipoles(F)

[ny,nx,nc] = im1.shape
xl, xu = 0, nx-1
yl1, yu1 = m1*xl+b1, m1*xu+b1
yl2, yu2 = m2*xl+b2, m2*xu+b2

from cv.util import colors, tcolors

# # draw figures
# pl.figure(1,(16,8))
# pl.subplot(1,2,1)
# pl.imshow(im1)
# pl.plot([xl,xu], [yl1,yu1], color = tcolors['blue'])
# pl.plot(pt1[:,0], pt1[:,1], color=colors['red'], marker='.', ls='None')
# pl.axis('off')
# pl.title('image 1', fontsize=8, weight='demibold')
# pl.subplot(1,2,2)
# pl.imshow(im2)
# pl.plot([xl,xu], [yl2,yu2], color = tcolors['blue'])
# pl.plot(pt2[:,0], pt2[:,1], color=colors['red'], marker='.', ls='None')
# pl.axis('off')
# pl.title('image 2', fontsize=8, weight='demibold')
# pl.show()
#
# # check that epipoles are at the intersection of epipolar lines
# pl.figure(1,(12,4))
# pl.subplot(1,2,1)
# xl, xu = 0, nx-1
# yl1, yu1 = m1*0+b1,     m1*e1[0]+b1
# yl2, yu2 = m2*e2[0]+b2, m2*(nx-1)+b2
# pl.plot([0,e1[0]], [yl1,yu1], color = tcolors['blue'])
# pl.plot(e1[0], e1[1], color=colors['red'], marker='.')
# pl.title('image 1', fontsize=8, weight='demibold')
# pl.subplot(1,2,2)
# pl.plot([e2[0],nx-1], [yl2,yu2], color = tcolors['blue'])
# pl.plot(e2[0], e2[1], color=colors['red'], marker='.')
# pl.title('image 2', fontsize=8, weight='demibold')
# pl.show()

cv.stereo.rectify_images(im1, e1)

