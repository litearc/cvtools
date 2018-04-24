# this demos concepts related to epipolar geometry

import cv
import matplotlib.pyplot as pl
import numpy as np
import scipy as sp
import sys
from PIL import Image as im

path = 'im/epipolar/'

# run epipolar geometry demo
m1 = cv.util.imread(path+'1.jpg')
p1 = cv.util.get_labeled_pts(path+'1-points.png').T # [{x,y} points]
m2 = cv.util.imread(path+'2.jpg')
p2 = cv.util.get_labeled_pts(path+'2-points.png').T # [{x,y} points]

# for rectification, use images with epipolar lines for visualization
m1 = cv.util.imread(path+'im1.png')
m2 = cv.util.imread(path+'im2.png')

cv.stereo.rectify_images(m1, p1, m2, p2)

