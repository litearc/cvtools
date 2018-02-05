# this module implements various computer vision algorithms. it's main purpose
# is to let me consolidate my understanding of these algorithms. there are
# almost certainly better implementations of these in other libraries, so
# there's no reason you should actually be using this.

import os, sys
import matplotlib as mp
import matplotlib.pyplot as pp
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

# utility functions ------------------------------------------------------------

def imshow(m):
  pp.imshow(m)
  pp.show()

def imread(f):
  return np.asarray(image.open(f)).astype(np.float64)/255

# ............................................................................

@jit
def calc_mu_icov(dat):
  
  # gets the mean and covariance of a set of rgb pixels.
  #
  # inputs ...................................................................
  # dat               input rgb values. (list) [r1,g1,b1,r2,g2,b2,...]
  #
  # outputs...................................................................
  # mu                mean rgb value. (3 x 1 matrix) [r;g;b]
  # cov               covariance matrix. (3 x 3 matrix)

  dat = np.matrix(np.reshape(dat,(-1,3))).T # [{r,g,b} pts]
  n = dat.shape[1] # num pts
  mu = np.mean(dat,1) # mean, matrix [r,g,b].T
  cov = (dat-mu)*(dat-mu).T/n # 3x3 covariance matrix
  return mu, cov


# bayesian matting -------------------------------------------------------------

# spec used for numba optimization
# bm_spec = [
#   ('img',     float32[:]),
#   ('tm',      float32[:]),
#   ('dvar',    float32),
#   ('lws',     int32),
#   ('overlap', float32),
#   ('npmin',   int32),
#   ('dwin',    float32),
#   ('a',       float32[:])
# ]

class Bayesian_Matting(object):
  """
  this class uses a maximum-likelihood (Bayesian) approach to compute the alpha
  values in the unknown tri-map pixels. it models the fg and bg, each, using a
  multivariate gaussian, and computes the fg, bg, and alpha values using an
  iterative approach until convergence. this is done over a shifting local
  window to improve likelihood of separating fg and bg.

  based on the method described in:
  Y. Chuang, B. Curless, D. Salesin, and R. Szeliski. A Bayesian approach to
  digital matting. In IEEE Computer Society Conference on Computer Vision and
  Pattern Recognition (CVPR), 2001.

  however, I used section 2.3 of "Computer Vision for Visual Effects" by Rich
  Radke for reference.
  """

  def __init__(self, img, tm, dvar=.4, lws=60, overlap=1.0/3, npmin=400):

    # inputs ...................................................................
    # img               input image. [y x {rgb}] (values b/w 0 and 1)
    # tm                trimap. [y x {rgb}] (0:bg, 1:fg, 0.5:unknown)
    # dvar              tunable parameter that reflects expected deviation from
    #                   matting assumption. (float)
    #                   log P(I|F,B,a) = -1/dvar * norm(I-(a*F+(1-a)*B)
    # lws               size of shifting local window is: lws x lws. (int)
    # overlap           fraction overlap b/w adjacent windows. (float)
    #                   (default = 1.0/3)
    # npmin             min num of fg/bg pixels we require to compute mean and
    #                   covariance matrix. (int)
    #
    # outputs ..................................................................
    # a                 alpha matte. [y x {rgb}]
    
    self.img = img
    self.tm = tm
    self.dvar = dvar
    self.lws = lws
    self.overlap = overlap
    self.dwin = int(self.lws*(1-overlap)) # num pixels for window to shift
    self.npmin = npmin
    self.a = self.main_driver()

  # ............................................................................

  def main_driver(self):

    # the main driver which shifts a local window over image, and for each
    # position, computes the alpha values for unknown pixels in the trimap. the
    # window is shifted in discrete increments, with some overlap. for unknown
    # pixels in overlap regions, the alpha estimates are averaged.

    # we add the alpha values estimated from each window into one array, and
    # track the number of values added so we can average at the end.
    [ny,nx,_] = self.img.shape
    navg = np.zeros((ny,nx,3))
    a = np.zeros((ny,nx,3))

    for iy in range(0,ny-1,self.dwin):
      for ix in range(0,nx-1,self.dwin):

        # look inside current window and see if there are enough (npmin) fg and
        # bg pixels to accurately compute mean and inverse covariance matrix. if
        # not, increase the size of window until there are.
        j, fgmu, fgicov, bgmu, bgicov = 0, np.nan, np.nan, np.nan, np.nan
        while True:

          iyl = np.maximum(0,iy-j*self.lws)
          iyu = np.minimum(ny-1,iy+(j+1)*self.lws-1)
          ixl = np.maximum(0,ix-j*self.lws)
          ixu = np.minimum(nx-1,ix+(j+1)*self.lws-1)
          ii = np.s_[iyl:iyu,ixl:ixu]
          
          if np.count_nonzero(self.tm[ii]==1)/3>=self.npmin and \
              np.any(np.isnan(fgmu)):
            fgmu, fgicov = calc_mu_icov(self.img[ii][self.tm[ii]==1])
            fgicov = sp.linalg.inv(fgicov)
          if np.count_nonzero(self.tm[ii]==0)/3>=self.npmin and \
              np.any(np.isnan(bgmu)):
            bgmu, bgicov = calc_mu_icov(self.img[ii][self.tm[ii]==0])
            bgicov = sp.linalg.inv(bgicov)
          if not np.any(np.isnan(fgmu)) and not np.any(np.isnan(bgmu)):
            break

          # if we can't find the min num of fg and bg pixels after increasing
          # window size many times, exit
          j += 1
          if j == 100:
            sys.exit()

        ii = np.s_[iy:iy+self.lws-1, ix:ix+self.lws-1,:]
        a[ii] += self.calc_alpha(self.img[ii], self.tm[ii], fgmu, fgicov,
            bgmu, bgicov)
        navg[ii] += 1

    a /= navg
    return a

  # ............................................................................

  def calc_alpha(self, img, tm, fgmu, fgicov, bgmu, bgicov):
    
    # iteratively compute the fg, bg, and alpha values in unknown # trimap
    # pixels within the current window.
    #
    # inputs ...................................................................
    # img               image to compute alpha values over. [y x {rgb}]
    # tm                associated trimap. [y x {rgb}]
    # fgmu              mean fg rgb value. (3 x 1 matrix) [r;g;b]
    # fgicov            inverse of fg covariance matrix. (3 x 3 matrix)
    # bgmu              mean bg rgb value. (3 x 1 matrix) [r;g;b]
    # bgicov            inverse of bg covariance matrix. (3 x 3 matrix)
    #
    # outputs ..................................................................
    # tmc               calculated trimap. [y x {rgb}]
  
    unp = tm[:,:,0]==.5 # unknown pixels
    tmc = tm[:,:,0] # calculated trimap
    [ny,nx,_] = img.shape
    for iy in range(ny):
      for ix in range(nx):
        if unp[iy,ix]: # is unknown pixel
          I = np.matrix(np.squeeze(img[iy,ix,:])).T
          [_,_,a] = self.compute_FGa(I,.5,fgmu,fgicov,bgmu,bgicov)
          tmc[iy,ix] = a

    # convert trimap from [ny nx] to [ny nx {rgb}]
    tmc = np.transpose(np.tile(tmc,[3,1,1]),[1,2,0])
    tmc[tmc<0] = 0 # bound trimap b/w 0 and 1
    tmc[tmc>1] = 1
    return tmc

  # ............................................................................

  def compute_FGa(self, I, a0, fgmu, fgicov, bgmu, bgicov, tol=.001,
      max_niter=40):
    
    # computes the fg, bg, and alpha value for a given pixel value. this is an
    # iterative maximum-likelihood procedure that repeats until convergence.
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
      a11 = np.matrix(fgicov+a**2/self.dvar*eye)
      a12 = np.matrix(a*(1-a)/self.dvar*eye)
      a21 = np.matrix(a*(1-a)/self.dvar*eye)
      a22 = np.matrix(bgicov+(1-a)**2/self.dvar*eye)
      A = np.vstack((np.hstack((a11,a12)), np.hstack((a21,a22))))

      b11 = np.matrix(fgicov*fgmu+a/self.dvar*I)
      b21 = np.matrix(bgicov*bgmu+(1-a)/self.dvar*I)
      b = np.vstack((b11,b21))

      # solve system using least-squares
      x = np.matrix(sp.linalg.lstsq(A,b)[0])

      # use computed fg and bg values to estimate alpha
      F,B = x[:3,0], x[3:,0]
      a = ((I-B).T*(F-B))[0,0] / ((F-B).T*(F-B))[0,0]

      # if F,B,a haven't changed much (converged), exit
      if (i > 0) and np.all(np.abs(x-xp)<tol) and np.abs(a-ap)<tol:
        break
      xp,ap = x,a # previous F,B,a values

    return F,B,a



# closed-form matting ----------------------------------------------------------

def matting_laplacian(I, r, eps):

  # gets the matting laplacian L, which is used in a minimization problem used
  # to obtain the alpha values. this is based on eq 2.36 in the Radke book.
  #
  # inputs .....................................................................
  # I                 image. [x y {r,g,b}]
  # r                 "radius" of square window. (int)
  # eps               regularization parameter. (float)
  #
  # outputs ....................................................................
  # a                 alpha values. [x y {rgb}]
  #
  
  [ny,nx,_] = I.shape
  n = nx*ny # num pixels in image
  w = (2*r+1)**2 # num pixels in window
  eye = np.matrix(np.eye(3))
  W = sp.sparse.dok_matrix((n,n))

  if verbose > 0:
    print('constructing matting laplacian:')
    widgets = [pb.Bar(':',left='[',right=']'), ' ', pb.ETA()]
    bar = pb.ProgressBar(widgets=widgets, max_value=ny-2*r, initial_value=0)

  # slide window over each position in image
  for wy in range(r,ny-r): # window y position
    if verbose > 0:
      bar.update(wy-r)
    for wx in range(r,nx-r): # window x position

      # part of window in image
      Iw = np.transpose(I[wy-r:wy+r+1,wx-r:wx+r+1,:],(1,0,2))

      # calculate mean and regularized inv covariance matrix of pixels in window
      mu, cov = calc_mu_icov(Iw)
      icov = sp.linalg.inv(cov+eps/w*eye)

      # get linear indices (row-major) of all pixels in window
      # k = np.ravel(np.arange(wy-r,wy+r+1)[:,np.newaxis]+
      #     np.arange(wx-r,wx+r+1)*ny)[np.newaxis]
      k = np.ravel(np.arange(wx-r,wx+r+1)[:,np.newaxis]*ny+
          np.arange(wy-r,wy+r+1))[np.newaxis]

      # construct a [w x 3] matrix containing rgb values of all pixels in window
      Ik = np.matrix(np.reshape(Iw,(w,3))).T

      # matrix where each element contains the value to add to an entry in L
      Wij = 1.0/w*(1+(Ik-mu).T*icov*(Ik-mu))

      # add to matting laplacian
      W[k.T,k] += Wij # (exploits fancy indexing + broadcasting)

  if verbose > 0:
    bar.finish()

  L = (sp.sparse.diags(np.ravel(np.sum(W,1)),0)-W).T
  return L

class Closed_Form_Matting(object):
  """
  this class implements a closed-form matting algorithm to estimate alpha values
  in unknown trimap pixels. it uses the color-line assumption (that alpha is
  linear wrt the image pixel value over a small window) and minimizes the error
  between the estimated alpha and the "linearized" alpha estimate.

  based on the method described in:
  A. Levin, D. Lischinski, and Y. Weiss. A closed-form solution to natural
  image matting. IEEE Transactions on Pattern Analysis and Machine
  Intelligence, 30(2):228-42, Feb. 2008.
  """

  def __init__(self, img, tm, lws=3, eps=1e-7, lam=100):

    # inputs ...................................................................
    # img               input image. [y x {rgb}] (values b/w 0 and 1)
    # tm                trimap. [y x {rgb}] (0:bg, 1:fg, 0.5:unknown)
    # lws               size of local window is: lws x lws. code assumed this
    #                   is an odd number. (int)
    # eps               regularization term to contrain linear coefficient for
    #                   pixel value. (float) (default = 1e-7)
    # lam               regularization term to contrain estimated alpha to
    #                   match trimap. (float) (default = 100)
    #
    # outputs ..................................................................
    # a                 alpha matte. [y x {rgb}]

    a = tm[:,:,0]
    k = ((a==0)|(a==1)).astype(np.int).T # mask for known alphas
    D = sp.sparse.diags(np.ravel(k),0)
    av = np.ravel(a.T)
    av[(av!=0)&(av!=1)] = 0
    L = matting_laplacian(img, int((lws-1)/2), eps*lws*lws)
    ac = sp.sparse.linalg.spsolve(L+lam*D,lam*av)
    ac = np.reshape(ac,k.shape).T
    ac[ac<0] = 0
    ac[ac>1] = 1
    self.a = ac[:,:,np.newaxis]



# ------------------------------------------------------------------------------

if __name__ == '__main__':

  # load data
  img = imread('inp.png') # raw image
  gta = imread('gt.png')  # ground-truth alpha
  tm = imread('tm.png')   # trimap
  tm[(tm!=0)&(tm!=1)] = .5
  if tm.ndim == 2:
    tm = np.transpose(np.tile(tm,(3,1,1)),((1,2,0)))

  # for test purposes - scale image down to reduce computation time
  fs = .25 # scale factor
  if fs != 1.0:
    [nx,ny,_] = img.shape
    sz = (int(nx*fs),int(ny*fs),3) # new size
    img = skimage.transform.resize(img,sz,mode='constant')
    gta = skimage.transform.resize(gta,sz,mode='constant')
    tm = skimage.transform.resize(tm,sz,mode='constant')
    tm[(tm!=0)&(tm!=1)] = .5

  # which algorithm to run?
  alg = 'closed-form'
  if alg == 'bayesian':
    m = Bayesian_Matting(img, tm.copy())
    pp.figure(1)
    pp.imshow(m.img)
    pp.figure(2)
    pp.imshow(m.a)
    pp.figure(3)
    pp.imshow(m.a*m.img)
    pp.figure(4)
    pp.imshow(tm)
    pp.show()
  if alg == 'closed-form':
    m = Closed_Form_Matting(img, tm.copy())
    pp.figure(1)
    pp.imshow(m.a*img)
    pp.show()

