from cv.util import colors, tcolors
import matplotlib.pyplot as pl
import numpy as np
from PIL import Image as im
import scipy as sp
import skimage.transform as tf
import pdb
import wc

# epipolar geometry -----------------------------------------------------------

def euc_to_hom(p):

  # converts points from euclidian to homogeneous coordinates.
  #
  # o = euc_to_hom(p)
  #
  # inputs ....................................................................
  # p                 points. [{x,y} points]
  #
  # outputs ...................................................................
  # o                 poitns. [{x,y,1} points]
  
  p = np.copy(p)
  n = p.shape[1]
  p = np.resize(p,(3,n))
  p[2,:] = 1
  return np.matrix(p)

def hom_to_euc(p):

  # converts points from homogeneous to euclidian coordinates.
  #
  # o = euc_to_hom(p)
  #
  # inputs ....................................................................
  # p                 points. [{x,y,1} points]
  #
  # outputs ...................................................................
  # o                 poitns. [{x,y} points]

  p = np.copy(np.array(p))
  p = p[:2,:]/p[2,:]
  return np.matrix(p)

def tform_pts(p, T):
  return hom_to_euc(T*euc_to_hom(p))

# -----------------------------------------------------------------------------

def get_null_vec(A):

  # finds the "null" vector `x` that minimizes ||A*x||. `A` should ideally be
  # a matrix with a rank-deficiency of 1, and `x` would be in its null space.
  # Due to noise, however, A will (likely) be non-singular, so instead we find
  # the "closest" vector `x`, i.e. minimizes ||A*x||.
  #
  # inputs ....................................................................
  # A                 matrix to find "null" vector for. (n x n matrix)
  #
  # outputs ...................................................................
  # x                 "null vector". (n x 1 vector)
  
  U,s,Vt = sp.linalg.svd(np.matrix(A))
  return Vt[-1,:].T

# -----------------------------------------------------------------------------

def cross_prod_matx(a):

  # given a vector 'a', returns a matrix 'A' such that cross(a,b) = A*b.
  # (where axb is the cross-product of a and b)

  return np.matrix([[0,-a[2],a[1]], [a[2],0,-a[0]], [-a[1],a[0],0]])

# -----------------------------------------------------------------------------

def tform_coefs(T):

  # given a projective transformation matrix 'T', returns the coefficients
  # needed to perform the transformation using Pillow's transform function.

  T = sp.linalg.inv(np.copy(T))
  T = T/T[2,2]
  return np.array(T).ravel()[:8]

# -----------------------------------------------------------------------------

def rotx(a):
  return np.matrix([[1,0,0],[0,np.cos(a),-np.sin(a)],[0,np.sin(a),np.cos(a)]])
def roty(a):
  return np.matrix([[np.cos(a),0,np.sin(a)],[0,1,0],[-np.sin(a),0,np.cos(a)]])
def rotz(a):
  return np.matrix([[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1]])

# -----------------------------------------------------------------------------

def get_fundamental_matrix(p1, p2):

  # gets the fundamental matrix associated with sets of points (correspondences)
  # in two images. requires at least 8 points (more is better).
  #
  # inputs .....................................................................
  # p1                points in image 1. [{x,y} points]
  # p2                points in image 2. [{x,y} points]
  #
  # outputs.....................................................................
  # F                 fundamental matrix. (3 x 3 matrix)

  # format data points to (3 x n) matrices
  n = p1.shape[1] # num points
  p1 = euc_to_hom(p1)
  p2 = euc_to_hom(p2)

  # first, normalize data so the data points in each image are centered about
  # the origin and have a mean distance from the origin of sqrt(2). this is
  # achieved by translation and scaling, captured in 3 x 3 matrices T1 and T2
  # (the matrices are used again later to de-normalize the fundamental matrix).
  p1,T1 = normalize_pts(p1)
  p2,T2 = normalize_pts(p2)

  x1, y1 = np.array(p1[0,:].T), np.array(p1[1,:].T)
  x2, y2 = np.array(p2[0,:].T), np.array(p2[1,:].T)
  A = np.hstack((x1*x2, y1*x2, x2, x1*y2, y1*y2, y2, x1, y1, np.ones((n,1))))

  # A should be singular but due to noise will (almost certainly) be
  # non-singular, so to find the F that minimizes ||A*F||_2, take the SVD of A
  # and look at the smallest right singular vector.
  F = get_null_vec(A).reshape((3,3))

  # F should also be singular, so again use the SVD, this time zero-ing out the
  # smallest singular value to make rank(F) == 2.
  U,s,Vt = sp.linalg.svd(F)
  F = np.matrix(U)*np.matrix(np.diag([s[0],s[1],0]))*np.matrix(Vt)
  F = T2.T*F*T1 # de-normalize
  return F

# -----------------------------------------------------------------------------

def normalize_pts(p):

  # normalizes the position of points so that they are centered about the
  # origin, i.e. have mean position (0,0), and have a mean distance of sqrt(2)
  # from the origin. these transformations can be given by a transformation
  # matrix.
  #
  # inputs .....................................................................
  # p                 point positions. [{x,y,1} points]
  #
  # outputs ....................................................................
  # p                 transformed positions. [{x,y,1} points]
  # T                 transformation matrix. (3 x 3 matrix)

  m = np.mean(p,1)
  d = np.mean(np.sqrt(np.sum(np.power(p,2),1))) # mean distance
  s = np.sqrt(2) # want points to have mean distance sqrt(2)
  T = np.matrix(np.diag([s/d,s/d,1]))*np.matrix(
      [[1,0,-m[0]],[0,1,-m[1]],[0,0,1]])
  p = T*p # apply transformation
  return p,T

# -----------------------------------------------------------------------------

def get_epipolar_lines(p1, p2, F=None):

  # finds the epipolar lines for a set of points and the fundamental matrix.
  # the lines are specified by the slope `m` and y-intercept `b` in the
  # equation: y = m*x+b
  #
  # inputs ....................................................................
  # p1                point positions in image 1. [{x,y} points]
  # p2                point positions in image 2. [{x,y} points]
  # F                 fundamental matrix. (3 x 3 matrix) (default = compute it)
  #
  # outputs ...................................................................
  # m1                slope for epipolar lines in image 1. (array)
  # b1                y-intercept for epipolar lines in image 1. (array)
  # m2                slope for epipolar lines in image 2. (array)
  # b2                y-intercept for epipolar lines in image 2. (array)

  if F is None:
    F = get_fundamental_matrix(p1, p2)

  # append column of 1s to positions
  n = p1.shape[1]
  p1 = euc_to_hom(p1)
  p2 = euc_to_hom(p2)

  # we assume the epipolar lines are not vertical (pretty safe assumption)
  e1 = np.array(p2.T*F) # e = [a,b,c], where a*x+b*y+c = 0 <-> y = -a/b*x-c/b
  e2 = np.array((F*p1).T) 

  # get epipolar lines
  m1,b1 = [-e1[:,0]/e1[:,1], -e1[:,2]/e1[:,1]]
  m2,b2 = [-e2[:,0]/e2[:,1], -e2[:,2]/e2[:,1]]
  return m1,b1,m2,b2

# -----------------------------------------------------------------------------

def get_epipoles(F):

  # finds the epipoles for two images with an associated fundamental matrix
  # `F`, given by x2*F*x1 = 0, where `x1` and `x2` are the positions of points
  # in homogeneous coordinates in images 1 and 2, respectively.
  #
  # based on section 5.4.1 in the Radke book.
  #
  # inputs ....................................................................
  # F                 fundamental matrix. (3 x 3 matrix)
  #
  # outputs ...................................................................
  # e1                position of epipole in image 1. (3 x 1 vector)
  # e2                position of epipole in image 2. (3 x 1 vector)

  ev,e1 = sp.linalg.eig(F)
  e1 = e1[:,np.argmin(np.abs(ev))]
  e1 /= e1[2]
  ev,e2 = sp.linalg.eig(F.T)
  e2 = e2[:,np.argmin(np.abs(ev))]
  e2 /= e2[2]
  return np.real(e1), np.real(e2)

# -----------------------------------------------------------------------------

def rectify_images(m1, p1, m2, p2):

  # rectifies two images so that the epipolar lines are horizontal, and
  # corresponding points in the images are found on the same horizontal line.
  #
  # see: Hartley and Zisserman. "Multiple view geometery in computer vision".
  # based on the method described in the CS231A Computer Vision course notes.
  #
  # im1, im2, H1, H2 = rectify_images(m1, p1, m2, p2)
  #
  # inputs ....................................................................
  # m1                image 1. [y x {rgb}]
  # p1                points in image 1. [{x,y} points]
  # m2                image 2. [y x {rgb}]
  # p2                points in image 2. [{x,y} points]
  # 
  # outputs ...................................................................
  # im1               rectified image 1. [y x {rgb}]
  # im2               rectified image 2. [y x {rgb}]
  # H1                transformation matrix for image 1. (3 x 3 matrix)
  # H2                transformation matrix for image 2. (3 x 3 matrix)

  p1 = euc_to_hom(p1)
  p2 = euc_to_hom(p2)
  F = get_fundamental_matrix(p1, p2)
  e1, e2 = get_epipoles(F)

  # first, rectify image 2
  ny, nx, _ = m2.shape
  T = np.matrix([[1,0,-nx/2], [0,1,-ny/2], [0,0,1]]) # center image
  ex, ey = e2[0]-nx/2, e2[1]-ny/2 # translated epipole
  a = -np.sign(ex)*np.arctan(ey/np.abs(ex))
  R = rotz(a) # rotation needed to move epipole to x-axis
  ex = (R*np.matrix([ex,ey,1]).T)[0]
  G = np.matrix([[1,0,0], [0,1,0], [-1/ex,0,1]]) # moves epipole to inf
  H2 = G*R*T
  
  c = tform_coefs(sp.linalg.inv(T)*H2)
  m2 = im.fromarray((np.copy(m2)*255).astype(np.uint8)) # scale to [0,255]
  m2 = np.array(m2.transform((nx,ny), im.PERSPECTIVE, c))

  # rectify image 1
  e2cp = cross_prod_matx(e2)
  M = e2cp*F+np.matrix(e2).T*np.matrix([1,1,1])

  # the two images are now rectified, but image 1 may be severely distorted
  # along the horizontal direction, so we apply a horizontal shear to minimize
  # the sos distance between point correspondences in the two images.
  p2t = np.array(H2*p2)
  p2t = np.matrix(p2t/p2t[2,:])
  p1t = np.array(H2*M*p1)
  p1t = np.matrix(p1t/p1t[2,:])
  a = np.linalg.lstsq(p1t.T, p2t.T[:,0], None)[0]
  HA = np.matrix([[a[0,0],a[1,0],a[2,0]], [0,1,0], [0,0,1]])

  c = tform_coefs(sp.linalg.inv(T)*HA*H2*M)
  m1 = im.fromarray((np.copy(m1)*255).astype(np.uint8)) # scale to [0,255]
  m1 = np.array(m1.transform((nx,ny), im.PERSPECTIVE, c))

  return m1, m2, sp.linalg.inv(T)*HA*H2*M, sp.linalg.inv(T)*H2

# -----------------------------------------------------------------------------

def view_morph(m1, p1, m2, p2):

  # performs a view morph of two input images and correspondences.
  #
  # inputs ....................................................................
  # m1                image 1. [y x {rgb}]
  # p1                points in image 1. [{x,y} points]
  # m2                image 2. [y x {rgb}]
  # p2                points in image 2. [{x,y} points]

  cc = np.random.permutation(np.arange(p1.shape[1]))
  pl.imshow(m1)
  pl.scatter(p1[0,:], p1[1,:], c=cc, marker='.')
  pl.show()
  pl.imshow(m2)
  pl.scatter(p2[0,:], p2[1,:], c=cc, marker='.')
  pl.show()

  # rectify images and correspondences
  m1r, m2r, H1, H2 = rectify_images(m1, p1, m2, p2)
  p1r = tform_pts(p1, H1)
  p2r = tform_pts(p2, H2)

  alpha = .3 # fraction the camera moves
  src, dst = np.array(p1r.T), np.array(p2r.T)

  # # ny,nx,_ = m1.shape
  # # src = np.array(np.append(src, [[0,0],[0,nx-1],[ny-1,0],[nx-1,ny-1]], axis=0))
  # # dst = np.array(np.append(dst, [[0,0],[0,nx-1],[ny-1,0],[nx-1,ny-1]], axis=0))
  #
  # # http://scikit-image.org/docs/dev/auto_examples/transform/plot_piecewise_affine.html

  tform = tf.PiecewiseAffineTransform()
  tform.estimate(src, dst)
  out = tf.warp(m1r, tform)
  pl.imshow(m1r)
  pl.show()
  pl.imshow(out)
  pl.show()

