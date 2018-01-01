# this module implements various computer vision algorithms. it's main purpose
# is to let me consolidate my understanding of these algorithms. there are
# almost certainly better implementations of these in other libraries, so
# there's no reason you should actually be using this.

import os
import imageio
import matplotlib.image as im
import matplotlib.pyplot as pp
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pylab as pl
import scipy.linalg as la
import scipy.misc

import pdb
import progressbar

# matting ......................................................................

# load data
num = '01'
img = im.imread('matting/inp/GT'+num+'.png') # raw image
tm = im.imread('matting/tm/1/GT'+num+'.png') # trimap
gta = im.imread('matting/gt/GT'+num+'.png')  # ground-truth alpha
# img = im.imread('matting/test/img.png') # raw image
# tm = im.imread('matting/test/tm.png') # trimap
colors = ('red','green','blue')

class BayesMatting(object):
  """
  this class uses a maximum-likelihood (Bayesian) approach to compute the alpha
  values in the "uncertain" tri-map region. the a-priori probabilities for the
  fg and bg are modeled using a single multivariate gaussian (each), and the
  a-priori alpha probability distribution is assumed to be constant.
  """

  def __init__(self, img, tm, dvar):
    self.img = img   # image  [y x {rgb}]
    self.tm = tm     # trimap [y x {rgb}]
    self.dvar = dvar # (sigma_d)^2
    self.fg_mu, self.fg_icov = self.get_params(img[tm==1])
    self.bg_mu, self.bg_icov = self.get_params(img[tm==0])
    self.a = self.calc_alpha()

  def calc_alpha(self):
    nun = np.count_nonzero(np.abs(self.tm[:,:,0]-.5)<.1) # num uncertain pixels
    pb = progressbar.ProgressBar(max_value=nun)
    tmc = self.tm[:,:,0]
    ii = 0
    [ny,nx,_] = img.shape
    for iy in range(ny):
      for ix in range(nx):
        if np.abs(tm[iy,ix,0]-.5) < .1: # in "uncertain" region of trimap
          I = np.matrix(np.squeeze(img[iy,ix,:])).T
          [F,B,a] = self.compute_FGa(I,.5)
          tmc[iy,ix] = a
          # set progressbar
          pb.update(ii)
          ii += 1
    tmc = np.transpose(np.tile(tmc,[3,1,1]),[1,2,0])
    tmc[tmc<0] = 0
    tmc[tmc>1] = 1
    return tmc

  def plot_rgb(self):
    # plot foreground, background, and uncertain pixels
    fg = np.reshape(self.img[self.tm==1],(-1,3))
    bg = np.reshape(self.img[self.tm==0],(-1,3))
    un = np.reshape(self.img[np.abs(self.tm-.5)<.1],(-1,3))
    ax = Axes3D(pp.gcf())
    ax.scatter(fg[:,0], fg[:,1], fg[:,2], 'b')
    ax.scatter(bg[:,0], bg[:,1], bg[:,2], 'r')
    ax.scatter(un[:,0], un[:,1], un[:,2], 'g')
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.set_zlim(0,1)
    ax.set_xlabel('red')
    ax.set_ylabel('green')
    ax.set_zlabel('blue')
    ax.legend(['foreground','background','uncertain'])
    pp.show()

  def get_params(self, dat):
    dat = np.matrix(np.reshape(dat,(-1,3))).T # [{r,g,b} pts]
    n = dat.shape[1] # num pts
    mu = np.mean(dat,1) # mean, matrix [r,g,b].T
    cov = dat*dat.T/n # 3x3 covariance matrix
    icov = la.inv(cov) # inverse of "
    return mu, icov

  def compute_FGa(self, I, a0, niter=10):
    a = a0
    eye = np.eye(3)
    for i in range(niter):
      a11 = np.matrix(self.fg_icov+a**2/self.dvar*eye)
      a12 = np.matrix(a*(1-a)/self.dvar*eye)
      a21 = np.matrix(a*(1-a)/self.dvar*eye)
      a22 = np.matrix(self.bg_icov+(1-a)**2/self.dvar*eye)
      A = np.vstack( (np.hstack((a11,a12)), np.hstack((a21,a22))) ) 

      b11 = np.matrix(self.fg_icov*self.fg_mu+a/self.dvar*I)
      b21 = np.matrix(self.bg_icov*self.bg_mu+(1-a)/self.dvar*I)
      b = np.vstack((b11,b21))

      x = np.matrix(la.lstsq(A,b)[0])
      F,B = x[:3,0], x[3:,0]
      a = ((I-B).T*(F-B))[0,0] / ((F-B).T*(F-B))[0,0]
    return F,B,a

bm = BayesMatting(img, tm, .4)
imageio.imwrite('/Users/hislam/Desktop/a.png', bm.a)
imageio.imwrite('/Users/hislam/Desktop/img.png', bm.a*bm.img)

