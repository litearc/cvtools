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

def imshow(m):
  pl.imshow(m)
  pl.show()

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
def calc_mu_icov(dat):
  
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

# ------------------------------------------------------------------------------

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
    [ny,nx] = ac.shape
    self.a = np.broadcast_to(ac[:,:,np.newaxis], (ny,nx,3))

# ------------------------------------------------------------------------------

# multi-res blending -----------------------------------------------------------

def downsample2(im, k):
  # blurs (via convolution) and downsamples image by a factor of 2
  im = sp.signal.convolve2d(im,k,'same')
  im = im[::2,::2]
  return im

def upsample2(im, k):
  # upsamples by a factor of 2, interpolates, and blurs image (via convolution)
  [ny,nx] = im.shape
  imu = np.zeros((ny*2,nx*2))
  imu[::2,::2] = im
  l = np.array([.5,1,.5])[np.newaxis]
  l = l.T*l # bilinear kernel
  imu = sp.signal.convolve2d(imu,l,'same')
  imu = sp.signal.convolve2d(imu,k,'same')
  return imu

def upsample2n(im, k, n):
  # upsamples image `n` times
  for i in range(n):
    im = upsample2(im,k)
  return im

def GL_pyramids(m, k, N):
  # constructs multi-scale Gaussian and Laplacian pyramids
  G = [m] # level 0
  for i in range(N): # levels 1 to N
    G.append(downsample2(G[-1],k))
  L = []
  for i in range(N): # levels 0 to N-1
    L.append(G[i]-sp.signal.convolve2d(G[i],k,'same'))
  L.append(G[N]) # level N
  return G,L

class Multi_Res_Blending(object):
  """
  this class implements multi-resolution blending of two images by blending
  lower frequencies across wide transition regions and higher frequencies across
  narrow transition regions. this produces a smoother blending with less
  noticeable transition between the images.

  I used section 3.1.2 of "Computer Vision for Visual Effects" by Rich Radke for
  reference.
  """

  def __init__(self, src, tgt, mask, N=4, lfilt=21, sigma=4):

    # inputs ...................................................................
    # src               source image. [y x {rgb}] (values b/w 0 and 1)
    # tgt               target image. [y x {rgb}] (values b/w 0 and 1)
    # mask              mask for source. [y x] (binary)
    # N                 num levels in multi-resolution pyramid. (int)
    #                   (default = 4)
    # lfilt             length of gaussian filter for convolution. convolution
    #                   kernel is: lfilt x lfilt. (int)
    # sigma             standard deviation for gaussian filter. (float)
    #                   (default = 4)
    #
    # outputs ..................................................................
    # I                 blended image. [y x {rgb}]
    #

    # compute gaussian convolution kernel
    g = sp.signal.gaussian(lfilt,sigma)
    g = g/np.sum(g)
    k = g[np.newaxis]
    k = k.T*k
    I = np.zeros(src.shape)
    Gm,Lm = GL_pyramids(mask,k,N)

    for ic in range(3): # rgb channels
      # compute gaussian pyramids for source, target, and mask
      Gs,Ls = GL_pyramids(src[:,:,ic],k,N)
      Gt,Lt = GL_pyramids(tgt[:,:,ic],k,N)
      Li = [] # laplacian of image at each scale
      for il in range(N+1):
        Li.append(upsample2n(Gm[il]*Ls[il]+(1-Gm[il])*Lt[il],k,il))
      I[:,:,ic] = Li[0]+Li[1]+Li[2]+Li[3]+Li[4]

    I[I<0],I[I>1] = 0,1
    self.I = I

# ------------------------------------------------------------------------------

# graph-cut compositing --------------------------------------------------------

class Graph_Cut_Compositing(object):
  """
  this class composits two aligned images by finding the minimum energy seam
  to stitch the images together, i.e. where the stitching is least noticeable.
  """

  def __init__(self, im, mask):
    
    # inputs ...................................................................
    # im                list of images. list of [x y {rgb}]
    # mask              masks for the images. list of [x y]
    #
    # outputs ..................................................................
    # m                 stitched image. [x y {rgb}]
 
    nim = len(im)
    if nim != 2:
      print('currently, only stitching 2 images is supported')
      return

    self.im = im
    self.mask = mask
    self.get_graph()

  def get_graph(self):

    [ny,nx,nc] = self.im[0].shape
    ip = lambda ix,iy : iy*nx+ix

    # constructing the graph one vertex/edge at a time is very slow. instead,
    # it is better to add all the vertices and edges using one call. with
    # igraph, vertices are automatically created if we pass a list of edges.
    
    # for the edges connecting the pixel vertices, we first create a 2D array
    # with each pixel index. for the vertical edges (e.g.), we stack two 2D
    # arrays (to create a 3D array with two layers): one with the first ny-1
    # rows, and one with the last ny-1 rows. then, at a given (x,y) position,
    # the indices in the two layers are the vertices the edge must connect.

    ii = np.arange(nx*ny).reshape((ny,nx)) # row-major order
    ev = np.concatenate((ii[1:,:].ravel()[None], ii[:-1,:].ravel()[None]))
    ev = list(map(tuple,ev.T)) # vertical edges
    eh = np.concatenate((ii[:,1:].ravel()[None], ii[:,:-1].ravel()[None]))
    eh = list(map(tuple,eh.T)) # horizontal edges

    # create edges from source to each source pixel. here, for the source
    # edges (e.g.), we create a 2D array with two columns. the 1st column
    # contains only the id of the source node, while the second column
    # contains the ids of each source pixel.

    id_src, id_tgt = nx*ny, nx*ny+1 # ids of src and tgt nodes
    isrc, itgt = self.mask[0]==1, self.mask[1]==1 # ids of src and tgt pixels
    nsrc, ntgt = np.sum(isrc), np.sum(itgt)
    
    es = (id_src*np.ones((nsrc,1),dtype=np.int32), ii[isrc].ravel()[None].T)
    es = list(map(tuple, np.hstack(es))) # source edges
    et = (id_tgt*np.ones((ntgt,1),dtype=np.int32), ii[itgt].ravel()[None].T)
    et = list(map(tuple, np.hstack(et))) # target edges

    # create graph
    e = ev+eh+es+et
    g = ig.Graph(e)

    # set weights for the edges. like with the vertices/edges, it is far more
    # efficient to set all the weights at once than to iterate over the edges.
    big_val = 1e6 # large weight b/w src/tgt nodes and src/tgt pixels

    # weight function is: ||im1(x1,y1)-im2(x1,y1)|| + ||im1(x2,y2)-im2(x2,y2)||
    # A. A. Efros and W. T. Freeman. Image quilting for texture synthesis and
    # transfer. In ACM SIGGRAPH (ACM Transactions on Graphics), 2001.
    mdiff = np.sum(np.square(self.im[0]-self.im[1]),2)
    wv = list((mdiff[1:,:]+mdiff[:-1,:]).ravel())
    wh = list((mdiff[:,1:]+mdiff[:,:-1]).ravel())
    ws, wt = nsrc*[big_val], ntgt*[big_val]
    w = wv+wh+ws+wt # all weights

    # do the min-cut!
    mcut = g.mincut(id_src, id_tgt, w)

    # combine images together
    mcomb = np.zeros((ny,nx,nc)) # combined image
    for l in range(2): # src, tgt
      ii = np.array(mcut.partition[l]) # pixels in src or tgt image
      i = np.where((ii==id_src)|(ii==id_tgt))[0] # indices of src/tgt nodes
      ii = np.delete(ii,i) # remove src/tgt nodes
      ix, iy = ii%nx, (ii/nx).astype(np.int32) # x,y indices of src image
      mcomb[iy,ix,:] = self.im[l][iy,ix,:]
    self.mcomb = mcomb

# ------------------------------------------------------------------------------

# graph-cut compositing --------------------------------------------------------

def get_fundamental_matrix(p1, p2, im1, im2):

  # gets the fundamental matrix associated with sets of points (correspondences)
  # in two images. requires at least 8 points (more is better).
  #
  # inputs .....................................................................
  # p1                points in image 1. [points {x,y}]
  # p2                points in image 2. [points {x,y}]
  #
  # outputs.....................................................................
  # F                 fundamental matrix. (3 x 3 matrix)

  # format data points to (3 x n) matrices
  n = np.shape(p1)[0] # num points
  p1, p2 = np.copy(p1).transpose((1,0)), np.copy(p2).transpose((1,0))
  p1, p2 = np.resize(p1,(3,n)), np.resize(p2,(3,n))
  p1[2,:], p2[2,:] = 1,1

  # first, normalize data so the data points in each image are centered about
  # the origin and have a mean distance from the origin of sqrt(2). this is
  # achieved by translation and scaling, captured in 3 x 3 matrices T1 and T2
  # (the matrices are used again later to de-normalize the fundamental matrix).
  m1, m2 = np.mean(p1,1), np.mean(p2,1) # mean (x,y) positions for images
  d1 = np.mean(np.sqrt(np.sum(p1**2,1))) # mean distance for image 1
  d2 = np.mean(np.sqrt(np.sum(p2**2,1))) # " for image 2
  s = np.sqrt(2)
  T1 = np.matrix(np.diag([s/d1,s/d1,1]))*np.matrix(
      [[1,0,-m1[0]],[0,1,-m1[1]],[0,0,1]])
  T2 = np.matrix(np.diag([s/d2,s/d2,1]))*np.matrix(
      [[1,0,-m2[0]],[0,1,-m2[1]],[0,0,1]])
  p1, p2 = T1*p1, T2*p2

  x1, y1 = np.array(p1[0,:].T), np.array(p1[1,:].T)
  x2, y2 = np.array(p2[0,:].T), np.array(p2[1,:].T)
  A = np.hstack((x1*x2, y1*x2, x2, x1*y2, y1*y2, y2, x1, y1, np.ones((n,1))))

  # A should be singular but due to noise will (almost certainly) be
  # non-singular, so to find the F that minimizes ||A*F||_2, take the SVD of A
  # and look at the smallest right singular vector.
  U,s,Vt = sp.linalg.svd(A)
  F = np.matrix(Vt[-1,:].reshape((3,3)))

  # F should also be singular, so again use the SVD, this time zero-ing out the
  # smallest singular value to make rank(F) == 2.
  U,s,Vt = sp.linalg.svd(F)
  F = np.matrix(U)*np.matrix(np.diag([s[0],s[1],0]))*np.matrix(Vt)
  F = T2.T*F*T1 # de-normalize
  return F

# ..............................................................................

def get_null_vec(A):
  U,s,Vt = sp.linalg.svd(np.matrix(A))
  return Vt[-1,:].T

# ..............................................................................

def draw_epipolar_lines(p1, p2, im1, im2, F):
  
  # append column of 1s to positions
  n = p1.shape[0]
  p1,p2 = np.matrix(np.copy(p1)), np.matrix(np.copy(p2))
  p1 = np.matrix(np.hstack((p1[:,0],p1[:,1],np.ones((n,1)))))
  p2 = np.matrix(np.hstack((p2[:,0],p2[:,1],np.ones((n,1)))))

  [ny,nx,nc] = im1.shape
  # we assume the epipolar lines are not vertical (pretty safe assumption)
  e1 = np.array(p2*F) # e = [a,b,c], where a*x+b*y+c = 0 <-> y = (-c-a*x)/b
  e2 = np.array((F*p1.T).T) 
  ep1 = get_null_vec(e1) # epipole of image 1
  ep2 = get_null_vec(e2)
  ep1 /= ep1[2]
  ep2 /= ep2[2]

  xl,xu = 0,ep1[0]
  y = lambda e,x: -(e[:,2]+e[:,0]*x)/e[:,1]
  yl1,yu1 = y(e1,xl), y(e1,xu)

  # check that epipole is at convergence of lines
  # pl.figure(1)
  # pl.plot([xl,xu], [yl1,yu1], color=colors['blue'])
  # pl.plot(ep1[0], ep1[1], 'ro')
  # pl.show()
  # return

  # get epipolar lines
  xl,xu = 0,nx-1
  y = lambda e,x: -(e[:,2]+e[:,0]*x)/e[:,1]
  yl1,yu1 = y(e1,xl), y(e1,xu)
  yl2,yu2 = y(e2,xl), y(e2,xu)
  
  pl.figure(1,(16,8))
  pl.subplot(1,2,1)
  pl.imshow(im1)
  pl.plot([xl,xu], [yl1,yu1], color = tcolors['blue'])
  pl.plot(p1[:,0], p1[:,1], color=colors['red'], marker='.', ls='None')
  pl.axis('off')
  pl.title('image 1', fontsize=8, weight='demibold')

  pl.subplot(1,2,2)
  pl.imshow(im2)
  pl.plot([xl,xu], [yl2,yu2], color = tcolors['blue'])
  pl.plot(p2[:,0], p2[:,1], color=colors['red'], marker='.', ls='None')
  pl.axis('off')
  pl.title('image 2', fontsize=8, weight='demibold')

  pl.show()

# ------------------------------------------------------------------------------

