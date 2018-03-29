import numpy as np
# from numba import jit
import progressbar as pb
import scipy as sp
import sys

# bayesian matting ------------------------------------------------------------

# @jit
def bayesian(im, tm, dvar=.4, lws=60, overlap=1.0/3, npmin=400):

  # This uses a maximum-likelihood (Bayesian) approach to compute the alpha
  # values in the unknown tri-map pixels. it models the fg and bg, each, using
  # a multivariate gaussian, and computes the fg, bg, and alpha values using an
  # iterative approach until convergence. this is done over a shifting local
  # window to improve likelihood of separating fg and bg.
  #
  # based on the method described in: Y. Chuang, B. Curless, D. Salesin, and R.
  # Szeliski. A Bayesian approach to digital matting. In IEEE Computer Society
  # Conference on Computer Vision and Pattern Recognition (CVPR), 2001.
  #
  # however, I used section 2.3 of "Computer Vision for Visual Effects" by Rich
  # Radke for reference.
  #
  # inputs ....................................................................
  # im                input image. [y x {rgb}] (values b/w 0 and 1)
  # tm                trimap. [y x {rgb}] (0:bg, 1:fg, 0.5:unknown)
  # dvar              tunable parameter that reflects expected deviation from
  #                   matting assumption. (float) (default = .4)
  #                   log P(I|F,B,a) = -1/dvar * norm(I-(a*F+(1-a)*B)
  # lws               size of shifting local window is: lws x lws. (int)
  #                   (default = 60)
  # overlap           fraction overlap b/w adjacent windows. (float)
  #                   (default = 1.0/3)
  # npmin             min num of fg/bg pixels we require to compute mean and
  #                   covariance matrix. (int) (default = 400)
  #
  # outputs ...................................................................
  # a                 alpha matte. [y x {rgb}]

  # this shifts a local window over the image, and for each position, computes
  # the alpha values for unknown pixels in the trimap. the window is shifted in
  # discrete increments, with some overlap. for unknown pixels in overlap
  # regions, the alpha estimates are averaged.

  # we add the alpha values estimated from each window into one 2D array, and
  # track the number of values added for each pixel for averaging at the end.
  [ny,nx,_] = im.shape
  navg = np.zeros((ny,nx,3))
  a = np.zeros((ny,nx,3))
  dwin = int(lws*(1-overlap)) # num pixels for window to shift

  # loop over windows
  for iy in range(0,ny-1,dwin):
    for ix in range(0,nx-1,dwin):

      # look inside current window and see if there are enough (npmin) fg and
      # bg pixels to accurately compute mean and inverse covariance matrix. if
      # not, increase the size of window until there are. the windows for the
      # fg and bg are independent, e.g. a small window may be needed to get
      # enough fg pixels, but a large window for enough bg pixels.
      j, fgmu, fgicov, bgmu, bgicov = 0, np.nan, np.nan, np.nan, np.nan
      while True:

        # lower and upper x and y pixel indices for the current window
        iyl = np.maximum(0,iy-j*lws)
        iyu = np.minimum(ny-1,iy+(j+1)*lws-1)
        ixl = np.maximum(0,ix-j*lws)
        ixu = np.minimum(nx-1,ix+(j+1)*lws-1)
        ii = np.s_[iyl:iyu,ixl:ixu]
        
        # check if there are enough fg pixels in current window
        if np.count_nonzero(tm[ii]==1)/3>=npmin and np.any(np.isnan(fgmu)):
          fgmu, fgicov = calc_mu_cov(im[ii][tm[ii]==1])
          fgicov = sp.linalg.inv(fgicov)

        # check if there are enough bg pixels in current window
        if np.count_nonzero(tm[ii]==0)/3>=npmin and np.any(np.isnan(bgmu)):
          bgmu, bgicov = calc_mu_cov(im[ii][tm[ii]==0])
          bgicov = sp.linalg.inv(bgicov)

        # if the window is large enough to have computed both fg and bg
        # estimates, break out of loop and compute alpha values
        if not np.any(np.isnan(fgmu)) and not np.any(np.isnan(bgmu)):
          break

        # if we can't find the min num of fg and bg pixels after increasing
        # window size many times, exit (user should consider reducing npmin)
        j += 1
        if j == 100:
          sys.exit()

      ii = np.s_[iy:iy+lws-1,ix:ix+lws-1,:] # indices for current window
      # accumulate alpha values from overlapping windows into the same array
      a[ii] += calc_alpha(im[ii], tm[ii], fgmu, fgicov, bgmu, bgicov, dvar)
      navg[ii] += 1 # keep track of num accumulations for each pixel

  # divide by num accumulations to get alpha value for each pixel
  a /= navg
  return a

# -----------------------------------------------------------------------------

# @jit
def calc_mu_cov(dat):
  
  # gets the mean and covariance of a set of rgb pixels.
  #
  # inputs ....................................................................
  # dat               input rgb values. (list) [r1,g1,b1,r2,g2,b2,...]
  #
  # outputs....................................................................
  # mu                mean rgb value. (3 x 1 matrix) [r;g;b]
  # cov               covariance matrix. (3 x 3 matrix)

  dat = np.matrix(np.reshape(dat,(-1,3))).T # [{r,g,b} pts]
  n = dat.shape[1] # num pts
  mu = np.mean(dat,1) # mean, matrix [r,g,b].T
  cov = (dat-mu)*(dat-mu).T/n # 3x3 covariance matrix
  return mu, cov

# -----------------------------------------------------------------------------

# @jit
def calc_alpha(im, tm, fgmu, fgicov, bgmu, bgicov, dvar):
  
  # iteratively compute the fg, bg, and alpha values in unknown # trimap
  # pixels within the current window.
  #
  # inputs ....................................................................
  # im                image to compute alpha values over. [y x {rgb}]
  # tm                associated trimap. [y x {rgb}]
  # fgmu              mean fg rgb value. (3 x 1 matrix) [r;g;b]
  # fgicov            inverse of fg covariance matrix. (3 x 3 matrix)
  # bgmu              mean bg rgb value. (3 x 1 matrix) [r;g;b]
  # bgicov            inverse of bg covariance matrix. (3 x 3 matrix)
  # dvar              tunable parameter that reflects expected deviation from
  #                   matting assumption. (float)
  #                   log P(I|F,B,a) = -1/dvar * norm(I-(a*F+(1-a)*B)
  #
  # outputs ...................................................................
  # tmc               calculated trimap. [y x {rgb}]

  unp = tm[:,:,0]==.5 # unknown pixels
  tmc = tm[:,:,0] # calculated trimap
  [ny,nx,_] = im.shape
  for iy in range(ny):
    for ix in range(nx):
      if unp[iy,ix]: # is unknown pixel
        I = np.matrix(np.squeeze(im[iy,ix,:])).T
        [_,_,a] = compute_FGa(I,.5,fgmu,fgicov,bgmu,bgicov, dvar)
        tmc[iy,ix] = a

  # convert trimap from [ny nx] to [ny nx {rgb}]
  tmc = np.transpose(np.tile(tmc,[3,1,1]),[1,2,0])
  tmc[tmc<0] = 0 # bound trimap b/w 0 and 1
  tmc[tmc>1] = 1
  return tmc

# -----------------------------------------------------------------------------

def compute_FGa(I, a0, fgmu, fgicov, bgmu, bgicov, dvar, tol=.001,
    max_niter=40):
  
  # computes the fg, bg, and alpha value for a given pixel value. this is an
  # iterative maximum-likelihood procedure that repeats until convergence.
  #
  # inputs ....................................................................
  # I                 image pixel value. (3 x 1 matrix) [r;g;b]
  # a0                initial alpha estimate. (float)
  # fgmu              mean fg rgb value. (3 x 1 matrix) [r;g;b]
  # fgicov            inverse of fg covariance matrix. (3 x 3 matrix)
  # bgmu              mean bg rgb value. (3 x 1 matrix) [r;g;b]
  # bgicov            inverse of bg covariance matrix. (3 x 3 matrix)
  # dvar              tunable parameter that reflects expected deviation from
  #                   matting assumption. (float)
  #                   log P(I|F,B,a) = -1/dvar * norm(I-(a*F+(1-a)*B)
  # tol               tolerance for convergence. converges when change in all
  #                   rgb elements of fg and bg and alpha value is < tol.
  #                   (float) (default = .001)
  # max_niter         max num iterations allowed. (int) (default = 40)
  #
  # outputs ...................................................................
  # F                 foreground pixel value. (3 x 1 matrix) [r;g;b]
  # B                 background pixel value. (3 x 1 matrix) [r;g;b]
  # a                 alpha value. (float)

  a = a0
  eye = np.eye(3)

  for i in range(max_niter):
    # set up linear system A*x = b, where A contains info about the fg and bg
    # distributions, x stores fg and bg pixel values (what we want to solve
    # for), and b contains the image pixel value (transformed).
    a11 = np.matrix(fgicov+a**2/dvar*eye)
    a12 = np.matrix(a*(1-a)/dvar*eye)
    a21 = np.matrix(a*(1-a)/dvar*eye)
    a22 = np.matrix(bgicov+(1-a)**2/dvar*eye)
    A = np.vstack((np.hstack((a11,a12)), np.hstack((a21,a22))))

    b11 = np.matrix(fgicov*fgmu+a/dvar*I)
    b21 = np.matrix(bgicov*bgmu+(1-a)/dvar*I)
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

# -----------------------------------------------------------------------------

# natural image matting -------------------------------------------------------

def natural(im, tm, lws=3, eps=1e-7, lam=100):

  # this class implements the natural image matting algorithm to estimate alpha
  # values in unknown trimap pixels. it uses the color-line assumption (that
  # alpha is linear wrt the image pixel value over a small window) and
  # minimizes the error between the estimated alpha and the "linearized" alpha
  # estimate.
  #
  # based on the method described in: A. Levin, D. Lischinski, and Y. Weiss. A
  # closed-form solution to natural image matting. IEEE Transactions on Pattern
  # Analysis and Machine Intelligence, 30(2):228-42, Feb. 2008.

  a = tm[:,:,0]
  k = ((a==0)|(a==1)).astype(np.int).T # mask for known alpha values
  D = sp.sparse.diags(np.ravel(k),0)
  av = np.ravel(a.T) # alpha values
  av[(av!=0)&(av!=1)] = 0

  # use matting laplacian to solve for unknown alpha values
  L = matting_laplacian(im, int((lws-1)/2), eps*lws*lws)
  ac = sp.sparse.linalg.spsolve(L+lam*D,lam*av)
  ac = np.reshape(ac,k.shape).T

  # some final processing of the alpha map
  ac[ac<0], ac[ac>1] = 0, 1 # make sure alpha values are between 0 and 1
  [ny,nx] = ac.shape
  a = np.broadcast_to(ac[:,:,np.newaxis], (ny,nx,3)) # make map [y x {rgb}]
  return a

# -----------------------------------------------------------------------------

def matting_laplacian(I, r, eps, verbose=0):

  # gets the matting laplacian L, which is used in a minimization problem used
  # to obtain the alpha values. this is based on eq 2.36 in the Radke book.
  #
  # inputs .....................................................................
  # I                 image. [x y {r,g,b}]
  # r                 "radius" of square window. (int)
  # eps               regularization parameter. (float)
  # verbose           show progressbar? (0 or 1) (default = 0)
  #
  # outputs ....................................................................
  # a                 alpha values. [x y {rgb}]
  
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
      mu, cov = calc_mu_cov(Iw)
      icov = sp.linalg.inv(cov+eps/w*eye)

      # get linear indices (row-major) of all pixels in window
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

# -----------------------------------------------------------------------------

