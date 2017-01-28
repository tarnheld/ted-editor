import numpy as np
import math as m
# matrix construction ########################################################

def identity():
  return np.array([[1., 0., 0., 0.],
	           [0., 1., 0., 0.],
	           [0., 0., 1., 0.],
	           [0., 0., 0., 1.]])

def scale(sx=1., sy=1., sz=1.):
  return np.array([[sx, 0., 0., 0.],
	           [0., sy, 0., 0.],
	           [0., 0., sz, 0.],
	           [0., 0., 0., 1.]])

def translate(tx=0., ty=0., tz=0.):
  return np.array([[1., 0., 0., tx],
	           [0., 1., 0., ty],
	           [0., 0., 1., tz],
	           [0., 0., 0., 1.]])

# rotation with angle a in radians about axis (x,y,z) 
def rotate(a=0., x=0., y=0., z=1.):
  s, c = m.sin(a), m.cos(a)
  h = m.sqrt(x*x + y*y + z*z)
  x, y, z = x/h, y/h, z/h
  sx, sy, sz = s*x, s*y, s*z
  oc = 1.-c
  return np.array([[oc*x*x+c,  oc*x*y-sz, oc*x*z+sy, 0.],
	           [oc*x*y+sz, oc*y*y+c,  oc*y*z-sx, 0.],
	           [oc*x*z-sy, oc*y*z+sx, oc*z*z+c,  0.],
	           [0.,        0.,        0.,        1.]])
# orthogonal projection
def ortho(l, r, b, t, n, f):
  w, h, d = r-l, t-b, f-n
  return np.array([[2./w, 0.,    0.,   -(r+l)/w],
	           [0.,   2./h,  0.,   -(t+b)/h],
	           [0.,   0.,   -2./d, -(f+n)/d],
	           [0.,   0.,    0.,    1.     ]])

# perspective projection, map frustum to unit cube
def frustum(l, r, b, t, n, f):
  w, h, d = r-l, t-b, f-n
  return np.array([[2.*n/w, 0.,      (r+l)/w,  0.      ],
	           [0.,     2.*n/h,  (t+b)/h,  0.      ],
	           [0.,     0.,     -(f+n)/d, -2.*f*n/d],
	           [0.,     0.,     -1.,       0.      ]])


# manipulation ###############################################################
def matrix(n, m, f=lambda i, j: 0.):
  I, J = range(n), range(m)
  return np.array([[f(i, j) for j in J] for i in I])

def size(A):
  return A.shape

def transpose(A):
  return np.transpose(A)

# opengl format
def column_major(A):
  return [float(a) for line in transpose(A) for a in line]

def exclude(A, i, j):
  return [R[:j]+R[j+1:] for R in A[:i]+A[i+1:]]

def top_left(A):
  n, m = size(A)
  return exclude(A, n-1, m-1)

# sum ########################################################################
def add(A, B):
  n, p = size(A)
  q, m = size(B)
  assert (n, p) == (q, m)
  return matrix(n, m, lambda i, j: A[i][j]+B[i][j])

def sub(A, B):
  n, p = size(A)
  q, m = size(B)
  assert (n, p) == (q, m)
  return matrix(n, m, lambda i, j: A[i][j]-B[i][j])
# product ####################################################################
def scalar(s, A):
  n, m = size(A)
  return matrix(n, m, lambda i, j: s*A[i][j])
def mul(A, B):
  return np.dot(A,B)
def product(A, *Bs):
  for B in Bs:
    A = np.dot(A,B)
  return A
def vapply(A,v):
  return np.dot(A,v)
# inverse ####################################################################
def det(A):
  return np.linalg.det(A)
def inverse(A):
  return np.linalg.inv(A)
# vector #####################################################################

# construction from list or tuples
def point(tpl):
  return np.array(tpl)
# vector from p1 to p0
def vector(p0, p1):
  return p0-p1

# homogenous vector from point p with weight w
def hom(p,w):
  return np.array([c*w for c in p] + [w]) if w else np.array([c for c in p] + [w])
# projection of homogenous vector
def proj(h):
  return np.array([h[i]/h[-1] if h[-1] else h[i] for i in range(0,len(h)-1)])
def cut(h):
  return np.array([h[i] for i in range(0,len(h)-1)])

# vector from coords
def coords(*args,dtype=float):
  return np.array(args,dtype=dtype)

# dot product of u and v: norm(u)*norm(v)*cos(angle between u and v)
def dot(u, v):
  return np.dot(u,v)
# squared norm of v
def norm2(v):
  return dot(v, v)
# norm of v, length of v
def norm(v):
  return m.sqrt(norm2(v))
# return unit vector in direction of v and length of v 
def unit_length(v):
  n = norm(v)
  return point(v)/n, n
# return unit vector in direction of v
def unit(v):
  n = norm(v)
  #import pdb ; pdb.set_trace()
  return v/n
# square matrix filled with v in each row
def vmatrix(v):
  return np.array([[vi] for vi in v])

# parallel part of x to y: the projection of x onto y
def para(x,y):
  return dot(x,y)/norm2(y) * y
# perpendicular part of x to y:
# x == para(x,y)+perp(x,y), dot(para(x,y),perp(x,y)) == 0
def perp(x,y):
  return x - para(x,y)
# reflection of x about y
def refl(x,y):
  return -x + 2*para(x,y)

# perpendicular vector to 2d vector x, counterclockwise
def perp2ccw(x):
  return coords(-x[1],x[0])

def perp2cw(x):
  return coords(x[1],-x[0])

