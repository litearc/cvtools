import igraph as ig
import numpy as np
import scipy as sp

# multi-resolution blending ---------------------------------------------------

def multi_res_blending(src, tgt, mask, N=4, lfilt=21, sigma=4):

  # this class implements multi-resolution blending of two images by blending
  # lower frequencies across wide transition regions and higher frequencies
  # across narrow transition regions. this produces a smoother blending with
  # less noticeable transition between the images.
  #
  # I used section 3.1.2 of "Computer Vision for Visual Effects" by Rich Radke
  # for reference.
  #
  # inputs ....................................................................
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
  # outputs ...................................................................
  # I                 blended image. [y x {rgb}]

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
  return I

# -----------------------------------------------------------------------------

def downsample2(im, k):
  
  # blurs (by convolution) and downsamples image by a factor of 2.
  #
  # inputs ....................................................................
  # im                input image. [y x]
  # k                 convolution kernel. [y x]
  #
  # outputs ...................................................................
  # im                output image with size [n/2y nx/2]. [y x]
  
  im = sp.signal.convolve2d(im,k,'same')
  im = im[::2,::2]
  return im

# -----------------------------------------------------------------------------

def upsample2(im, k):

  # upsamples by a factor of 2, interpolates, and blurs image (by convolution).
  #
  # inputs ....................................................................
  # im                input image. [y x]
  # k                 convolution kernel. [y x]
  #
  # outputs ...................................................................
  # im                output image with size [2*ny 2*nx]. [y x]

  [ny,nx] = im.shape
  imu = np.zeros((ny*2,nx*2))
  imu[::2,::2] = im
  l = np.array([.5,1,.5])[np.newaxis]
  l = l.T*l # bilinear kernel
  imu = sp.signal.convolve2d(imu,l,'same') # bilinear interpolation
  imu = sp.signal.convolve2d(imu,k,'same')
  return imu

# -----------------------------------------------------------------------------

def upsample2n(im, k, n):

  # upsamples image `n` times.
  #
  # inputs ....................................................................
  # im                input image. [y x]
  # k                 convolution kernel. [y x]
  # n                 number of times to upsample. (int)
  #
  # outputs ...................................................................
  # im                output image with size [ny*(2^n) nx*(2^n)]. [y x]

  for i in range(n):
    im = upsample2(im,k)
  return im

# -----------------------------------------------------------------------------

def GL_pyramids(m, k, N):

  # constructs multi-scale Gaussian and Laplacian pyramids.
  #
  # inputs ....................................................................
  # m                 input image. [y x]
  # k                 convolution kernel. [y x]
  # N                 the number of levels in Gaussian and Laplacian pyramids
  #                   is N+1. (int)
  #
  # outputs ...................................................................
  # G                 Gaussian pyramid. (list of successively smaller images)
  # L                 Laplacian pyramid. (list of successively smaller images)

  G = [m] # level 0
  for i in range(N): # levels 1 to N
    G.append(downsample2(G[-1],k))
  L = []
  for i in range(N): # levels 0 to N-1
    L.append(G[i]-sp.signal.convolve2d(G[i],k,'same'))
  L.append(G[N]) # level N
  return G,L

# -----------------------------------------------------------------------------

# graph-cut compositing -------------------------------------------------------

def graph_cut_compositing(im, mask):

  [ny,nx,nc] = im[0].shape
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
  isrc, itgt = mask[0]==1, mask[1]==1 # ids of src and tgt pixels
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
  mdiff = np.sum(np.square(im[0]-im[1]),2)
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
    mcomb[iy,ix,:] = im[l][iy,ix,:]
  return mcomb

# -----------------------------------------------------------------------------

