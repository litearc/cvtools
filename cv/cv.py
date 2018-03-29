# this module implements various computer vision algorithms. it's main purpose
# is to let me consolidate my understanding of these algorithms. there are
# almost certainly better implementations of these in other libraries, so
# there's no reason you should actually be using this.

import os, sys
import cv2
import igraph as ig
import matplotlib as mp
import matplotlib.pyplot as pl
import numpy as np
from PIL import Image as image
import scipy as sp
import skimage as sk
import skimage.transform
import progressbar as pb

from numba import jit, jitclass
from numba import int32, float32

# global variables
verbose = 1

colors = {
  'blue'      : '#348abd',
  'purple'    : '#7a68a6',
  'red'       : '#a60628',
  'green'     : '#467821',
  'pink'      : '#cf4457',
  'turquoise' : '#188487',
  'orange'    : '#e24a33'
}
tcolors = {
  'blue'      : '#348abd80',
  'purple'    : '#7a68a680',
  'red'       : '#a6062880',
  'green'     : '#46782180',
  'pink'      : '#cf445780',
  'turquoise' : '#18848780',
  'orange'    : '#e24a3380'
}

# utility functions ------------------------------------------------------------

def imread(f):
  return np.asarray(image.open(f)).astype(np.float64)/255

def fread(f):
  fp = open(f,'r')
  l = fp.readlines()
  nv,nc = len(l),len(l[0].split())
  o = np.zeros((nv,nc))
  for i in range(nv):
    o[i,:] = l[i].split()
  return o

def getnz(f):
  m = imread(f)[:,:,0]
  o = np.nonzero(m)
  return np.vstack((o[1],o[0]))

def get_labeled_pts(f):
  m = np.asarray(image.open(f))[:,:,0]
  p = np.nonzero(m)
  n = len(p[0])
  o = np.zeros((n,2))
  for i in range(n):
    l = m[p[0][i],p[1][i]]
    o[l-1,0],o[l-1,1] = p[1][i],p[0][i]
  return o

# ..............................................................................

@jit
def calc_mu_cov(dat):
  
  # gets the mean and covariance of a set of rgb pixels.
  #
  # inputs .....................................................................
  # dat               input rgb values. (list) [r1,g1,b1,r2,g2,b2,...]
  #
  # outputs.....................................................................
  # mu                mean rgb value. (3 x 1 matrix) [r;g;b]
  # cov               covariance matrix. (3 x 3 matrix)

  dat = np.matrix(np.reshape(dat,(-1,3))).T # [{r,g,b} pts]
  n = dat.shape[1] # num pts
  mu = np.mean(dat,1) # mean, matrix [r,g,b].T
  cov = (dat-mu)*(dat-mu).T/n # 3x3 covariance matrix
  return mu, cov

# ------------------------------------------------------------------------------

