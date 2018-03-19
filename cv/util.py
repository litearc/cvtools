import numpy as np
from PIL import Image as image

# -----------------------------------------------------------------------------

def imread(f):

  # loads an image as a numpy array (for some reason, I was getting slightly
  # different (incorrect) results with other methods).
  #
  # inputs ....................................................................
  # f                 file path. (string)
  #
  # outputs ...................................................................
  # m                 image as numpy array. [y x] or [y x {rgb}]
  
  return np.asarray(image.open(f)).astype(np.float64)/255

# -----------------------------------------------------------------------------

def get_labeled_pts(f):

  # this takes an image, and creates a 2D array where each row corresponds to
  # a non-zero pixel in the red channel of the image. the rows are ordered
  # by the value of the non-zero pixel, i.e. the 1st row corresponds to the
  # pixel with value 1, the 2nd row to the pixel with value 2, etc... the
  # implementation assumes that there is only one pixel for any non-zero value,
  # and the non-zero values are continuous (1,2,3,...).
  #
  # inputs ....................................................................
  # f                 file path. (string)
  #
  # outputs ...................................................................
  # o                 output array. [n 2] where n is the number of non-zero
  #                   entries in the image.

  m = np.asarray(image.open(f))[:,:,0]
  p = np.nonzero(m)
  n = len(p[0])
  o = np.zeros((n,2))
  for i in range(n):
    l = m[p[0][i],p[1][i]]
    o[l-1,0],o[l-1,1] = p[1][i],p[0][i]
  return o

# -----------------------------------------------------------------------------

