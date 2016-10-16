import linalg as la

def lerp(t, a, b):
  return (1-t)*a + t*b

def bezier2(t,ps):
  umt = 1-t
  return umt*umt*ps[0] + 2*umt*t*ps[1] + t*t * ps[2]

def bezier2_d(t,ps):
  umt = 1-t
  return 2*umt*(ps[1] - ps[0]) + 2*t*(ps[2] - ps[1])


def bezier2_dr(t,ps):
  # use quotient rule
  cut = [la.cut(ps[0]),la.cut(ps[1]),la.cut(ps[2])]
  xt = bezier2(t,cut)
  dxt = bezier2_d(t,cut)
  w = [ps[0][2],ps[1][2],ps[2][2]]
  wt = bezier2(t,w)
  dwt = bezier2_d(t,w)
  return (dxt*wt - dwt*xt)/(wt*wt)


def arc(p0,p1,p2):
  w = la.dot(la.unit(p1 - p0), la.unit(p2 - p0))
  return [la.hom(p0,1),la.hom(p1,w),la.hom(p2,1)]


#A = arc(la.coords(0,0),la.coords(1,1),la.coords(0,2))
A = [la.coords(1,0,1),la.coords(0,4,0),la.coords(5,0,1)]
p  = bezier2(0.5,A)
t  = bezier2_dr(0.5,A)
print (p,t)
print (la.proj(p))
