import matplotlib.pyplot as pl
import numpy as np
from PIL import Image as im
import scipy as sp
import skimage.transform as tf

# epipolar geometry -----------------------------------------------------------

def get_fundamental_matrix(p1, p2):

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
  # p                 point positions. [points {x,y}]
  #
  # outputs ....................................................................
  # p                 transformed positions. [points {x,y}]
  # T                 transformation matrix. (3 x 3 matrix)

  m = np.mean(p,1)
  d = np.mean(np.sqrt(np.sum(p**2,1)))
  s = np.sqrt(2)
  T = np.matrix(np.diag([s/d,s/d,1]))*np.matrix(
      [[1,0,-m[0]],[0,1,-m[1]],[0,0,1]])
  p = T*p
  return p,T

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

def get_epipolar_lines(p1, p2, F=None):

  # finds the epipolar lines for a set of points and the fundamental matrix.
  # the lines are specified by the slope `m` and y-intercept `b` in the
  # equation: y = m*x+b
  #
  # inputs ....................................................................
  # p1                point positions in image 1. [points {x,y}]
  # p2                point positions in image 2. [points {x,y}]
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
  n = p1.shape[0]
  p1,p2 = np.matrix(np.copy(p1)), np.matrix(np.copy(p2))
  p1 = np.matrix(np.hstack((p1[:,0],p1[:,1],np.ones((n,1)))))
  p2 = np.matrix(np.hstack((p2[:,0],p2[:,1],np.ones((n,1)))))

  # we assume the epipolar lines are not vertical (pretty safe assumption)
  e1 = np.array(p2*F) # e = [a,b,c], where a*x+b*y+c = 0 <-> y = -a/b*x -c/b
  e2 = np.array((F*p1.T).T) 

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
  return e1,e2

# -----------------------------------------------------------------------------

def rectify_images(m1, e1):

  m1 = m1*255 # scale from [0,1] to [0,255] (need for PIL)
  ex,ey = e1[0],e1[1] # location of epipole

  # translate image so image center is at center of rotation
  [ny,nx,_] = m1.shape
  T = np.matrix([[1,0,-nx/2], [0,1,-ny/2], [0,0,1]])

  # rotate image so epipole is at y = 0
  ex,ey = e1[0]-nx/2,e1[1]-ny/2
  d = np.sqrt(ex*ex+ey*ey)
  a = -np.sign(ex)*np.arctan(ey/np.abs(ex))
  R = np.matrix([[np.cos(a),-np.sin(a),0], [np.sin(a),np.cos(a),0], [0,0,1]])

  e = R*T*np.matrix([ex,ey,1]).T # new location of epipole
  G = np.matrix([ [1,0,0], [0,1,0], [-1/e[0],0,1] ]) # moves epipole to inf

  H = sp.linalg.inv(T)*G*R*T
  c = np.array(H).ravel()[:8]

  im1 = im.fromarray(m1.astype(np.uint8))
  im1 = np.array(im1.transform((nx,ny), im.PERSPECTIVE, c))
  pl.imshow(im1)
  pl.show()

  # a = np.pi/16
  # R = np.matrix([ [np.cos(a),-np.sin(a),0], [np.sin(a),np.cos(a),0], [0,0,1] ])
  # print(R)
  #
  # H = sp.linalg.inv(T)*R*T
  # f = tf.ProjectiveTransform(H)
  # # m1t = tf.warp(m1,f.inverse)
  # m1t = tf.warp(m1,f)
  # pl.imshow(m1t)
  # pl.show()

  # T = np.matrix([[1,0,-800], [0,1,-600], [0,0,1]])
  # e = np.pi/16
  # R = np.matrix([[np.cos(e),-np.sin(e),0], [np.sin(e),np.cos(e),0], [0,0,1]])
  # f = tf.ProjectiveTransform(sp.linalg.inv(T)*R*T)
  # m1t = tf.warp(m1,f.inverse)
  # pl.imshow(m1t)
  # pl.show()
  
  # [ny,nx,_] = m1.shape
  # # T = np.matrix([[1,0,-nx/2], [0,1,-ny/2], [0,0,1]])
  # # f = tf.ProjectiveTransform(T)
  #
  # a = [-1,1][e1[0]>0]
  # c = np.sqrt(e1[0]**2+e1[1]**2)
  # # R = np.matrix([[a*e1[0]/c,a*e1[1]/c,0], [-a*e1[0]/c,a*e1[1]/c,0], [0,0,1]])
  # # e1new = R*T*np.matrix(e1[:,None])
  # # G = np.matrix([[1,0,0], [0,1,0], [-1/e1new[0],0,1]])
  # # H = sp.linalg.inv(T)*G*R*T
  #
  # pl.imshow(m1)
  # pl.plot(1200,800,'ro')
  # pl.show()
  #
  # # f = tf.ProjectiveTransform(H)
  # t = np.pi/32
  # R = np.matrix([[np.cos(t),-np.sin(t),0], [np.sin(t),np.cos(t),0], [0,0,1]])
  # T = np.matrix([[1,0,-800], [0,1,-600], [0,0,1]])
  # f = tf.ProjectiveTransform(sp.linalg.inv(T)*R*T)
  #
  # pt = np.matrix([1200,800,1]).T
  # newpt = (sp.linalg.inv(T)*R*T)*pt
  # print(sp.linalg.inv(T)*R*T*pt)
  #
  # mrot = tf.warp(m1, f)
  # pl.imshow(mrot)
  # pl.plot(newpt[0], newpt[1], 'ro')
  # pl.show()

# -----------------------------------------------------------------------------

