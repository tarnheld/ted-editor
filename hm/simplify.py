import numpy as np

# adapeted from http://mourner.github.io/simplify-js/

def dot(u, v):
  return np.dot(u,v)

def norm2(v):
  return dot(v, v)
def norm(v):
  return np.sqrt(norm2(v))
def unit_length(v):
  n = norm(v)
  return point(v)/n, n
def unit(v):
  n = norm(v)
  return  point(v)/n

def para(x,y):
  return np.dot(x,y)/norm2(y) * y
def perp(x,y):
  return x - para(x,y)
def refl(x,y):
  return -x + 2*para(x,y)

def segdist(p,p1,p2):
    v1 = p2 - p1
    v2 = p  - p1
    nv2 = norm2(v1)
    e = p1
    if nv2 > 0.00001:
        t = dot(v2,v1)/nv2
        if t > 1:
            e = p2
        elif t > 0:
            e = v1 * t
    v3 = p - e
    
    return norm2(v3)

def simplifyRadDist(points,sqTol):
    prev = points[0]
    new = [prev]
    p = None
    for i in range(1,len(points)):
        p = points[i]
        if norm2(p-prev) > sqTol:
            new.append(p)
            prev = p
    if prev is not p:
        new.append(p)
    return new

def simplifyDPStep(points, first,last,sqTol,simplified):
    maxSqDist = sqTol
    maxi = None
    for i in range(first+1,last):
        sqDist = segdist(points[i],points[first],points[last])
        if sqDist > maxSqDist:
            maxi = i
            maxSqDist = sqDist
    if maxSqDist > sqTol:
        if maxi - first > 1:
            simplifyDPStep(points,first,maxi,sqTol,simplified)
        simplified.append(points[maxi])
        if last - maxi > 1:
            simplifyDPStep(points,maxi,last,sqTol,simplified)

def simplifyDP(points,sqTol):
    last = len(points)-1
    simplified = [points[0]]
    simplifyDPStep(points,0,last,sqTol,simplified)
    simplified.append(points[last])
    return simplified

def simplify(points, tol, hq):
    if len(points) < 2:
        return points
    sqTol = tol*tol
    if hq:
        points = simplifyRadDist(points,sqTol)
    points = simplifyDP(points,sqTol)
    return np.array(points)
