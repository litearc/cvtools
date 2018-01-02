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

# bayesian matting -------------------------------------------------------------

# load data
# num = '01'
# img = im.imread('matting/inp/GT'+num+'.png') # raw image
# tm = im.imread('matting/tm/1/GT'+num+'.png') # trimap
# gta = im.imread('matting/gt/GT'+num+'.png')  # ground-truth alpha
img = im.imread('matting/test/img.png') # raw image
tm = im.imread('matting/test/tm.png') # trimap
colors = ('red','green','blue')

class Bayesian_Matting(object):
  """
  this class uses a maximum-likelihood (Bayesian) approach to compute the alpha
  values in the unknown tri-map pixels. it models the fg and bg, each, using a
  multivariate gaussian, and computes the fg, bg, and alpha values using an
  iterative approach until convergence. this is done over a shifting local
  window to improve likelihood of separating of fg and bg.

  based on the method described in:
  Y.-Y. Chuang, B. Curless, D. Salesin, and R. Szeliski. A Bayesian approach to
  digital matting. In IEEE Computer Society Conference on Computer Vision and
  Pattern Recognition (CVPR), 2001.

  however, I used section 2.3 of "Computer Vision for Visual Effects" by Rich
  Radke for reference.
  """

  def __init__(self, img, tm, dvar, lws):

    # inputs ...................................................................
    # img               input image. [y x {rgb}]
    # trimap            trimap. [y x {rgb}]
    # dvar              tunable parameter that reflects expected deviation from
    #                   matting assumption. (float)
    #                   log P(I|F,B,a) = -1/dvar * norm(I-(a*F+(1-a)*B)
    # lws               size of shifting local window is: lws x lws. (int)
    
    self.img = img
    self.tm = tm
    self.dvar = dvar
    self.fg_mu, self.fg_icov = self.get_mu_icov(img[tm==1])
    self.bg_mu, self.bg_icov = self.get_mu_icov(img[tm==0])
    self.a = self.calc_alpha()

  # ............................................................................

  def calc_alpha(self):
    
    # the main driver which shifts a local window over image, and for each
    # position computes the mean and covariance matrix for fg and bg (obtained
    # from trimap), and iteratively computes the fg, bg, and alpha values in
    # unknown trimap pixels within the current window.
    #
    # outputs ..................................................................
    # tmc               calculated trimap. [y x {rgb}]
    
    # indicate progress with bar
    unp = np.abs(self.tm[:,:,0]-.5)<.1 # unknown pixels
    nun = np.count_nonzero(unp) # num unknown pixels
    pb = progressbar.ProgressBar(max_value=nun)
    ii = 0 # num unknown pixels processed so far

    tmc = self.tm[:,:,0] # calculated trimap
    [ny,nx,_] = img.shape
    for iy in range(ny):
      for ix in range(nx):
        if unp[iy,ix]: # is unknown pixel
          I = np.matrix(np.squeeze(img[iy,ix,:])).T
          [F,B,a] = self.compute_FGa(I,.5)
          tmc[iy,ix] = a
          # set progressbar
          pb.update(ii)
          ii += 1

    # convert trimap from [ny nx] to [ny nx {rgb}]
    tmc = np.transpose(np.tile(tmc,[3,1,1]),[1,2,0])
    tmc[tmc<0] = 0 # bound trimap b/w 0 and 1
    tmc[tmc>1] = 1
    return tmc

  # ............................................................................

  def get_mu_icov(self, dat):
    
    # gets the mean and covariance of a set of rgb pixels.
    #
    # inputs ...................................................................
    # dat               input rgb values. (list) [r1,g1,b1,r2,g2,b2,...]
    #
    # outputs...................................................................
    # mu                mean rgb value. (3 x 1 matrix) [r;g;b]
    # icov              inverse of covariance matrix. (3 x 3 matrix)

    dat = np.matrix(np.reshape(dat,(-1,3))).T # [{r,g,b} pts]
    n = dat.shape[1] # num pts
    mu = np.mean(dat,1) # mean, matrix [r,g,b].T
    cov = dat*dat.T/n # 3x3 covariance matrix
    icov = la.inv(cov) # inverse of "
    return mu, icov

  # ............................................................................

  def compute_FGa(self, I, a0, tol=.001, max_niter=20):
    
    # computes the fg, bg, and alpha value for a given pixel value. this is an
    # iterative maximum-likelihood procedure that goes until convergence.
    #
    # inputs ...................................................................
    # I                 image pixel value. (3 x 1 matrix) [r;g;b]
    # a0                initial alpha estimate. (float)
    # tol               tolerance for convergence. converges when change in rgb
    #                   elements of fg and bg and alpha value is < tol.
    #                   (float) (default = .001)
    # max_niter         max num iterations allowed. (int) (default = 20)
    #
    # outputs ..................................................................
    # F                 foreground pixel value. (3 x 1 matrix) [r;g;b]
    # B                 background pixel value. (3 x 1 matrix) [r;g;b]
    # a                 alpha value. (float)

    a = a0
    eye = np.eye(3)

    for i in range(max_niter):
      # set up linear system A*x = b, where A contains info about the fg and bg
      # distributions, x stores fg and bg pixel values (what we want to solve
      # for), and b contains the image pixel value (transformed).
      a11 = np.matrix(self.fg_icov+a**2/self.dvar*eye)
      a12 = np.matrix(a*(1-a)/self.dvar*eye)
      a21 = np.matrix(a*(1-a)/self.dvar*eye)
      a22 = np.matrix(self.bg_icov+(1-a)**2/self.dvar*eye)
      A = np.vstack((np.hstack((a11,a12)), np.hstack((a21,a22))))

      b11 = np.matrix(self.fg_icov*self.fg_mu+a/self.dvar*I)
      b21 = np.matrix(self.bg_icov*self.bg_mu+(1-a)/self.dvar*I)
      b = np.vstack((b11,b21))

      # solve system using least-squares
      x = np.matrix(la.lstsq(A,b)[0])

      # use computed fg and bg values to estimate alpha
      F,B = x[:3,0], x[3:,0]
      a = ((I-B).T*(F-B))[0,0] / ((F-B).T*(F-B))[0,0]

      # if F,B,a haven't changed much (converged), exit
      if (i > 0) and np.all(np.abs(x-xp)<tol) and np.abs(a-ap)<tol:
        break
      xp,ap = x,a # previous F,B,a values

    return F,B,a

# ------------------------------------------------------------------------------

bm = Bayesian_Matting(img, tm, .4, 100)

# imageio.imwrite('/Users/hislam/Desktop/a.png', bm.a)
# imageio.imwrite('/Users/hislam/Desktop/img.png', bm.a*bm.img)

