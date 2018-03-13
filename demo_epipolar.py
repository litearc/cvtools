# this demos concepts related to epipolar geometry

import cvtools
import matplotlib.pyplot as pl
import numpy as np
from PIL import Image as image
import skimage.transform

path = 'im/epipolar/'

# run epipolar geometry demo
im1 = cvtools.imread(path+'1.jpg')
pt1 = cvtools.get_labeled_pts(path+'1-points.png')
im2 = cvtools.imread(path+'2.jpg')
pt2 = cvtools.get_labeled_pts(path+'2-points.png')

F = cvtools.get_fundamental_matrix(pt1, pt2, im1, im2)
cvtools.draw_epipolar_lines(pt1, pt2, im1, im2, F)

