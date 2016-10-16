#adapted from
#https://www.researchgate.net/publication/228966598_Optimal_Single_Biarc_Fitting_and_its_Applications
#calculates the biarc going through p0 and p1 with tangents t0 and t1
#respective. returns the homogeneous control points of a bezier curve
#defining the circular arcs. 
def biarc_h(p0, t0, p1, t1, r):
  V  = p1 - p0

  rt0 = la.refl(t0,V)
  x = la.dot(rt0,t1)
  print(x)
  # #if x < -0.9:
  # #r = 1/abs((1+x))

  U  = r*t0 + t1
  
  nc = la.norm2(V)
  b = 2*la.dot(V,U)
  a = 2*r*(1 - la.dot(t0,t1))
  
  eps = 1/2**16
  
  if a < eps:
    if abs(b) < eps: # U=V, beta infinite, second circle is a semicircle
      print("abs(b=",b,") < eps")
      J = (p0+p1)/2
      r1 = m.sqrt(nc)/4
      h = [la.hom(r1*t0,0),la.hom(J,1),la.hom(-r1*t1,0)]
      sys.stdout.flush()
      return h
    beta = nc / b
    print("abs(a=",a,") < eps, beta=",beta)
  else:
    D = b*b + 4*a*nc
    beta = (-b + m.sqrt(D)) / (2*a)
    #beta = 2*nc/(b + m.sqrt(D))
    
  alpha = beta * r
  if abs(b) < eps and x < 0: # U=V, beta infinite, second circle is a semicircle
    print("panic!")

    # parametrize circle of joints
    pt0 = la.refl(t0,V)
    pt1 = la.refl(t1,V)
    tt0 = t0+pt0
    tt1 = t1+pt1

    nVcosa2 = la.dot(tt0,V)
    d = nc/2*Nvcosa2

    JP = p0 + d*t0
    cosa2 = nVcosa2 / m.sqrt(nc)
    J  = la.proj(bezier2(0.5,[la.hom(p0,1),la.hom(JP,cosa2),la.hom(p1,1)]))
    Jt = pt0

    ch1 = J-p0
    ch2 = p1-J
    
    alpha = la.norm2(ch1)/2/cos(la.dot(t0,ch1))
    beta  = la.norm2(ch2)/2/cos(la.dot(t1,ch2))
  else:
    alpha = r * beta
    ab = alpha + beta 
    c1 = p0 + alpha * t0
    c3 = p1 - beta  * t1

  print(alpha,beta,a,b,nc)
    
  c2 = (beta / ab)  * c1 + (alpha / ab) * c3

  w1 = la.dot(t0, la.unit(c2 - p0))
  w2 = la.dot(t1, la.unit(p1 - c2))
  sys.stdout.flush()
  return la.hom(c1,w1),la.hom(c2,1),la.hom(c3,w2)

