import sys
from enum import Enum
from copy import copy,deepcopy
import pickle
import itertools as it
import colorsys
import math as m

import tkinter as tk
from tkinter import filedialog

import CanvasX as CX
from PIL import ImageTk, Image, ImageFilter

import numpy as np

import linalg as la

from fsm import FSM
import read_ted as ted

import raildefs

sys.path.append("eledit")
import eledit as eed 

import upload_ted as ut

##############################################################################
def lerp(t, a, b):
  return (1-t)*a + t*b
def bezier2(t,ps):
  """ evaluate quadratic bezier curve at t given three points in list ps """
  umt = 1-t
  return umt*umt*ps[0] + 2*umt*t*ps[1] + t*t*ps[2]
def bezier2_d(t,ps):
  """ evaluate derivative of quadratic bezier curve at t given
  coordinates of three points in list ps """
  umt = 1-t
  return 2*umt*(ps[1] - ps[0]) + 2*t*(ps[2] - ps[1])
def bezier2_dr(t,ps):
  """ evaluate derivative of rational quadratic bezier curve at t given
  homogeneous coordinates of three points in list ps """
  # rational curve, use quotient rule to get derivative
  cut = [la.cut(ps[0]),la.cut(ps[1]),la.cut(ps[2])]
  xt  = bezier2(t,cut)
  dxt = bezier2_d(t,cut)
  w   = [ps[0][-1],ps[1][-1],ps[2][-1]] # weights
  wt  = bezier2(t,w)
  dwt = bezier2_d(t,w)
  return (dxt*wt - dwt*xt)/(wt*wt)
def bezier2_h(t, hs):
  """ evaluate quadratic bezier curve at t and project result """
  x = bezier2(t,hs)
  return la.proj(x)
def circle2_h(t, hs):
  """ evaluate quadratic bezier circle at t from -1 to 1, if t < 0 use the opposite arc"""
  x = bezier2(abs(t),[hs[0],hs[1] if t > 0 else -hs[1], hs[2]])
  return la.proj(x)
def bezier2_tangent_h(t,hs):
  """ return unit vector in the direction of the tangent of rational quadratic bezier curve at t """
  dx = bezier2_dr(t,hs)
  return unit(dx)
def bezier_circle_from_two_points_and_tangent(p0,p1,t0):
  """ return homogeneous control points of rational quadratic bezier
  circle going through p0 and p1 with tangent t0 at p0 """
  chord = p1 - p0
  ct,cl = la.unit_length(chord)

  cosa2 = la.dot(t0,ct) # cosine of half the angle between tangents at p0 and p1

  # gotta love homogeneous coordinates, normal calculation of midpoint looks like this:
  # pm = p0 + cl/(2*cosa2) * t
  # notice the division through cosa2 and the special cases needed when it goes to zero...
  ph = p0*cosa2 + cl/2 * t0
  return [la.hom(p0,1),la.coords(ph[0],ph[1],cosa2),la.hom(p1,1)]
def circle_center_from_two_points_and_tangent(p0,p1,t0):
  chord = p1 - p0
  ct,cl = la.unit_length(chord)
  n     = la.perp2ccw(t0)
  sina2 = la.dot(n,ct)  # sine of half the angle between tangents at p0 and p1
  ch = p0*sina2 + cl/2 * n
  return la.coords(ch[0],ch[1],sina2)
def switch_circle_segment(hs):
  """given a rational quadratic bezier circle, switch the segment that is
  evaluated for t from 0 to 1. this enables evaluation of the whole circle"""
  hs[1] = -hs[1]
def biarc_joint_circle_tangent(p0,t0,p1,t1):
  """find the tangent of the joint circle of the biarc at p0"""
  V   = p1 - p0
  eps = 1/2**12

  pt1 = la.refl(t1,V) # reflect t1 about chord
  d0  = t0 + pt1
  if la.norm2(d0) < eps:
    #if la.dot(t0,t1) > 0: # joint circle is a line
    #  d0 = t0
    #elif la.dot(t0,t1) < 0: # joint circle is minimal
      d0 = la.perp2ccw(t0)

  tJ0 = la.unit(d0)
  return tJ0
def biarc_joint_circle(p0, t0, p1, t1):
  """returns the circle of all biarc joint points"""
  tJ0 = biarc_joint_circle_tangent(p0,t0,p1,t1)
  return bezier_circle_from_two_points_and_tangent(p0,p1,tJ0) # joint circle
def biarc_joint_point(Jb,r):
  """returns the joint point of a biarc given the joint circle and a
  parameter between -1 and 1"""
  return circle2_h(r,Jb)
def biarc_tangent_at_joint_point(J,p0,t0):
  """returns the tangent of both joining biarc circle segments at joint
  point J, given start point p0 and start tangent t0"""
  return la.refl(t0,J - p0)
def biarc_h(p0, t0, p1, t1, r):
  """construct biarc from p0 with tangent t0 to p1 with tangent t1, at
   joint circle rational bezier parameter r. returns both circles and
   the joint circle"""
  Jb = biarc_joint_circle(p0,t0,p1,t1)
  J  = biarc_joint_point(Jb,r)
  Jt = biarc_tangent_at_joint_point(J,p0,t0)
  C1 = bezier_circle_from_two_points_and_tangent(p0,J,t0)
  C2 = bezier_circle_from_two_points_and_tangent(J,p1,Jt)
  return C1,C2,Jb

def circleparam(p0,p1,t0):
  """find circle parameters for circle given by a chord from
   p0 to p1 and tangent t0 at p0. returns center, signed curvature k,
   segment angle subtended, and the arclength

  """
  chord = p1-p0
  ut0 = la.unit(t0)
  n = la.perp2ccw(ut0)
  chordsinha = la.dot(chord,n)
  chordcosha = la.dot(chord,ut0)
  chord2 = la.norm2(chord)
  signed_curvature = 2 * chordsinha / chord2
  lc = m.sqrt(chord2)

  ha = 0
  length = 0
  
  eps = 2**-16
  if abs(signed_curvature) > eps:
    center = p0 + n * (1 / signed_curvature)
    ha = m.acos(chordcosha/lc)
    length = 2 * ha / abs(signed_curvature)
  else:
    center = p0
    length = lc
    signed_curvature = 0

  return center, signed_curvature, 2*ha, length
def bc_arclen_t(s,a):
  """return bezier parameter of rational bezier circle for arclength s
  and total segment angle a (angle between first and last bezier control point)"""
  if a > 0:
    ha = a/2
    return m.sin(ha*s)/(m.sin(ha*s)+m.sin(ha*(1-s)))
  else:
    return s
def bezier_circle_parameter(p0,p1,t0,pc):
  """given point on circle pc and bezier start and end points p0 and p1
  and tangent t0 at p0 return bezier parameter of p"""
  t = la.norm(pc-p0)/(la.norm(pc-p0)+la.norm(pc-p1))
  # first dot determines if pc is on short (>0) or long segment,
  # second determines if joint circle t > 0 is on short segment (>0) or long
  chord = p0 - p1
  f1 = -la.dot(pc - p0, pc - p1)
  f2 = -la.dot(t0,chord)
  si = f1*f2
  eps = 2**-16

  if abs(si) < eps:
    # chord is 2*radius, check vector perpendicular to chord and joint circle
    # tangent at p0
    tp = la.perp(pc,chord)
    si = la.dot(tp,t0)

  if (si < 0):
    t = -t
  return t
def point_on_circle(p0,p1,t0,p):
  """finds point on circle nearest to p, returns point on circle,
  corresponding parameter value t and segment angle"""
  ch = circle_center_from_two_points_and_tangent(p0, p1, t0)

  eps = 2**-16

  if abs(ch[2]) < eps: # circle is line
    top = p - p0
    chord = p1 - p0
    pc = la.para(top,chord)
    t = la.dot(pc,chord)/la.norm2(chord)
    return p0 + pc, t, 0
  else:
    c = la.proj(ch)
    tp = la.unit(p - c)
    r = la.norm(c - p0)
    pc = c + r * tp

    t = bezier_circle_parameter(p0,p1,t0,pc)
    return pc, t, la.dot(la.unit(t0),la.unit(pc-p0))
def offset_circle(hs,o):
  t0 = la.unit(bezier2_dr(0,  hs)) # tangent at p0
  t1 = la.unit(bezier2_dr(0.5,hs)) # tangent at p1
  t2 = la.unit(bezier2_dr(1,  hs)) # tangent at p2
  pt0 = la.hom(la.perp2ccw(t0),0)  # offset direction at p0
  pt1 = la.hom(la.perp2ccw(t1),0)  # offset direction at p1
  pt2 = la.hom(la.perp2ccw(t2),0)  # offset direction at p2
  return [hs[0] + o*pt0, hs[1] + o*pt1, hs[2] + o*pt2]
def transform_bezier_circle(hs,xform):
  for i,h in enumerate(hs):
    th = la.vapply(xform,la.coords(h[0],h[1],0,h[2]))
    hs[i] = la.coords(th[0],th[1],th[3])
  return hs
def transform_point(p,xform):
  tp = la.vapply(xform,la.coords(p[0],p[1],0,1))
  return la.coords(tp[0],tp[1])
def transform_vector(v,xform):
  tv = la.vapply(xform,la.coords(v[0],v[1],0,0))
  return la.coords(tv[0],tv[1])
###########################################################################
def interpol_bi(im, x, y):
  x = np.asarray(x)
  y = np.asarray(y)
  
  x0 = np.floor(x).astype(int)
  x1 = x0 + 1
  y0 = np.floor(y).astype(int)
  y1 = y0 + 1
  
  x0 = np.clip(x0, 0, im.shape[1]-1);
  x1 = np.clip(x1, 0, im.shape[1]-1);
  y0 = np.clip(y0, 0, im.shape[0]-1);
  y1 = np.clip(y1, 0, im.shape[0]-1);
  
  Ia = im[ y0, x0 ]
  Ib = im[ y1, x0 ]
  Ic = im[ y0, x1 ]
  Id = im[ y1, x1 ]
  
  wa = (x1-x) * (y1-y)
  wb = (x1-x) * (y-y0)
  wc = (x-x0) * (y1-y)
  wd = (x-x0) * (y-y0)
  
  return wa*Ia + wb*Ib + wc*Ic + wd*Id
###########################################################################
class Biarc:
  def __init__(self, p0, t0, p1, t1, r):
    self.p0 = p0
    self.p1 = p1
    self.t0 = la.unit(t0)
    self.t1 = la.unit(t1)
    self.r = r
    self.JC = biarc_joint_circle(p0,t0,p1,t1)
    self.J  = biarc_joint_point(self.JC,r)
    self.Jt = biarc_tangent_at_joint_point(self.J,p0,t0)
    self.h1 = bezier_circle_from_two_points_and_tangent(self.p0,self.J, self.t0)
    self.h2 = bezier_circle_from_two_points_and_tangent(self.J, self.p1,self.Jt)
    self.cp1 = circleparam(self.p0, self.J,  self.t0)
    self.cp2 = circleparam(self.J,  self.p1, self.Jt)
  def joint(self):
    return self.J

  def circleparameters(self):
    c1,k1,a1,l1 = self.cp1
    c2,k2,a2,l2 = self.cp2

    return c1,k1,a1,l1,c2,k2,a2,l2

  def length(self):
    c1,k1,a1,l1 = self.cp1
    c2,k2,a2,l2 = self.cp2
    return l1+l2

  def offset(self,o):
    b = deepcopy(self)
    b.h1 = offset_circle(b.h1,o)
    b.h2 = offset_circle(b.h2,o)
    b.JC = offset_circle(b.JC,o)

    b.p0 = bezier2_h(0,b.h1)
    b.J  = bezier2_h(1,b.h1)
    b.p1 = bezier2_h(1,b.h2)

    b.cp1 = circleparam(b.p0,b.J,b.t0)
    b.cp2 = circleparam(b.J,b.p1,b.Jt)
    return b
  def transform(self,xform):
    self.p0 = transform_point(self.p0,xform)
    self.p1 = transform_point(self.p1,xform)
    self.t0 = la.unit(transform_vector(self.t0,xform))
    self.t1 = la.unit(transform_vector(self.t1,xform))
    self.JC = transform_bezier_circle(self.JC,xform)
    self.J  = transform_point(self.J,xform)
    self.Jt = la.unit(transform_vector(self.Jt,xform))
    self.h1 = transform_bezier_circle(self.h1,xform)
    self.h2 = transform_bezier_circle(self.h2,xform)
    self.cp1 = circleparam(self.p0, self.J,  self.t0)
    self.cp2 = circleparam(self.J,  self.p1, self.Jt)
  def t_and_hs_of_rsl(self,rsl):
    c1,k1,a1,l1,c2,k2,a2,l2 = self.circleparameters()
    s = rsl*(l1+l2)
    rs = 0
    if s < l1:
      rs = s/l1
      t = bc_arclen_t(rs,a1)
      return t,self.h1
    else:
      rs = (s-l1)/l2
      t = bc_arclen_t(rs,a2)
      return t,self.h2
  def eval(self,rsl):
    """eval biarc at rsl from 0 to 1, meaning arc length/total biarc length"""
    t,hs = self.t_and_hs_of_rsl(rsl)
    return bezier2_h(t, hs)
  def eval_t(self,rsl):
    """eval biarc tangent at rsl from 0 to 1, meaning arc length/total biarc length"""
    t,hs = self.t_and_hs_of_rsl(rsl)
    return la.unit(bezier2_dr(t, hs))
  def evalJ(self,t):
    """eval joint circle from -1 to 1"""
    return circle2_h(t, self.JC)
  def dist_sqr_at_l(self,p):
    """return squared distance and arc length at nearest point to p on biarc"""
    p1,t1,cha1 = point_on_circle(self.p0,self.J,self.t0,p)
    p2,t2,cha2 = point_on_circle(self.J,self.p1,self.Jt,p)
    c1,k1,a1,l1,c2,k2,a2,l2 = self.circleparameters()
    mind,minl = None,None
    if t1 > 0 and t1 < 1:
      mind = la.norm2(p1-p)
      aa1 = 2*m.acos(cha1)
      minl = aa1 / abs(k1) if k1 != 0 else t1*l1
    if t2 > 0 and t2 < 1 and (not mind or la.norm2(p2-p) < mind):
      mind = la.norm2(p2-p)
      aa2 = 2*m.acos(cha2)
      minl = l1 + (aa2 / abs(k2) if k2 != 0 else t2*l2)
    return mind,minl
###########################################################################
def line_seg_distance_sqr_and_length(p0,p1,p):
  chord = p0 - p1
  top   = p0 - p
  pa    = la.para(top, chord)
  if la.dot(pa,chord) < 0:
    return la.norm2(p - p0),0  # distance to first point
  elif la.norm2(chord) < la.norm2(pa):
    return la.norm2(p - p1),la.norm(chord)  # distance to last point
  else:
    return la.norm2(la.perp(top, chord)),la.norm(pa) # distance to line segment
###########################################################################
class Straight:
  def __init__(self, p0, p1):
    self.p0 = p0
    self.p1 = p1
  def length(self):
    return la.norm(self.p1 - self.p0)
  def offset(self,o):
    pt = la.perp2ccw(la.unit(self.p1 - self.p0))
    return Straight(self.p0 + pt*o,self.p1 + pt*o)
  def eval(self,t):
    return lerp(t,self.p0,self.p1)
  def eval_t(self,t):
    return  la.unit(self.p1 - self.p0)

  def dist_sqr_at_l(self,p):
    "return squared distance to p and length of segment at nearest point"
    return line_seg_distance_sqr_and_length(self.p0,self.p1,p)
  def transform(self,xform):
    self.p0 = transform_point(self.p0,xform)
    self.p1 = transform_point(self.p1,xform)
  TedCp  = ted.ted_data_tuple("cp",ted.cp)
  TedSeg = ted.ted_data_tuple("segment",ted.segment)
  def circleparameters(self):
    c1,k1,a1,l1 = (0,0),0,0,la.norm(self.p0-self.p1)
    c2,k2,a2,l2 = (0,0),0,0,0

    return c1,k1,a1,l1,c2,k2,a2,l2
def offsetLine(segment,ts,te,o,n):
  so = segment.offset(o)
  cas = []
  nn = max(n,2)
  for j in range(nn):
    t = lerp(j/(nn-1),ts,te)
    p = so.eval(t)
    cas.append(p)
  return cas

def offsetPolygon(segment,ts,te,o1,o2,n):
  return offsetLine(segment,ts,te,o1,n) + offsetLine(segment,te,ts,o2,n)

###########################################################################
class CCPoint:
  def __init__(self, point = la.coords(0,0), tangent=None):
    self.point = point
    self.tangent = tangent
###########################################################################
class SegType(Enum):
  Straight = 1
  Biarc = 2
###########################################################################
def canvas_create_circle(canvas,p,r,**kw):
  "convenience canvas draw routine"
  return canvas.create_oval([p[0]-r,p[1]-r,p[0]+r,p[1]+r],**kw)

class Banking:
  def __init__(self,angle=0,prev_len=0,next_len=0):
    self.angle = angle
    self.prev_len = prev_len
    self.next_len = next_len

###########################################################################
class CCSegment:
  def __init__(self, p1, p2, type = SegType.Straight, width=8, biarc_r=0.5):
    self.ps      = p1
    self.pe      = p2
    self.type    = type
    self.width   = width
    self.biarc_r = biarc_r
    self.banking = [Banking(),Banking()]
    self.heights = [[],[]]

    self.setup()

  def setup(self):
    if self.type is SegType.Straight:
      self.ps.tangent = self.pe.tangent = la.unit(self.pe.point - self.ps.point)
      self.seg = Straight(self.ps.point,self.pe.point)
      self.np  = 2
    else:
      if self.pe.tangent is None:
        self.pe.tangent = la.unit(la.refl(self.ps.tangent,
                                          self.ps.point - self.pe.point))

      self.seg = Biarc(self.ps.point, self.ps.tangent,
                       self.pe.point, self.pe.tangent,
                       self.biarc_r)
      self.np  = 32
    self.poly = offsetPolygon(self.seg,0,1,-self.width/2,self.width/2,self.np)
    self.heights_need_update = True
    #print("seg setup update!")
    #sys.stdout.flush()
  def shapeparameters(self):
    """ returns end points, lengths, curvature, center of circle of subsegments"""
    divl = 4.0
    mindiv = 6
    if self.type is SegType.Straight:
      l = self.seg.length()
      div = max(mindiv,int(m.ceil(l/divl)))
      return [(self.pe.point,l,div,0,la.coords(0,0))]
    else:
      c1,k1,a1,l1,c2,k2,a2,l2 = self.seg.circleparameters()
      div1 = max(mindiv,int(m.ceil(l1/divl)))
      div2 = max(mindiv,int(m.ceil(l2/divl)))
      return [(self.seg.J,l1,div1,k1,c1),(self.pe.point,l2,div2,k2,c2)]
  def num_subsegments(self):
    return len(self.shapeparameters())
  def set_heights(self,heights):
    self.heights = heights
    self.heights_need_update = False
    #print("seg height update!")
    sys.stdout.flush()
  def calc_heights(self):
    def heightAt(p,ex,hme,hmw,hmh):
      ip = (p+ex)/(2*ex) * la.coords(hmw,hmh)
      hs = interpol_bi(hmm,ip[0],ip[1])
      return hs * hme[1] + (1-hs) * hme[0]
    hm = np.load("hm/andalusia.npz")
    hme = hm["extents"]
    hmm = hm["heightmap"]
    hmw = hmm.shape[0]
    hmh = hmm.shape[1]
    ex = 3499.99975586

    sps = self.shapeparameters()
    
    hs = heightAt(self.ps.point,ex,hme,hmw,hmh)
    for j,sp in enumerate(sps):
      hgts = []
      ep,l,div,k,center = sp
      he = heightAt(ep,ex,hme,hmw,hmh)
      for d in range(div):
        hgts.append(hs * (div - d)/div + he * d/div)
      self.heights[j] = hgts
      hs = he
    #print("seg height calc!")
    sys.stdout.flush()
    self.heights_need_update = False
    #exhgts = hgts[-6:-1] + hgts + hgts[0:6]
    #print(len(hgts),len(exhgts))
    #ex = len(exhgts) - len(hgts)
    #for i in range(len(hgts)):
    #  ma = exhgts[i:i+ex]
    #  #print("before",hgts[i],ex)
    #  hgts[i] = sum(ma)/len(ma)
    #  #print("after",hgts[i])
  def reverse(self):
    # reverse endpoints from point array
    self.ps,self.pe   = self.pe,self.ps
    # reverse biarc parameter
    self.biarc_r = 1 - self.biarc_r
    # TODO: banking data reverse
    self.setup()
  def transform(self,xform):
    self.seg.transform(xform)
    for i,p in enumerate(self.poly):
      self.poly[i] = transform_point(p,xform)
  def draw(self,canvas,cids=None,**kw):
    if not cids: cids = []
    cids.append(canvas.create_polygon([(x[0],x[1]) for x in self.poly],**kw))
    return cids
  def drawText(self,canvas,cids=None,**kw):
    if not cids: cids = []
    #cids.extend([canvas_create_circle(canvas,x,5) for x in self.poly])
    if self.type is SegType.Straight:
      h = self.seg.eval(0.5)
      l = self.seg.length()
      txt = "{:.0f}".format(l)
      cids.append(canvas.create_text([h[0],h[1]], text=txt, tags="segment"))
    else:
      c1,k1,a1,l1,c2,k2,a2,l2 = self.seg.circleparameters()
      h1 = self.seg.eval(0.5*l1/(l1+l2))
      h2 = self.seg.eval((l1+0.5*l2)/(l1+l2))
      #h1 = c1
      #h2 = c2

      if abs(k1) < 1/300:
        s1text = "\n{:.0f}".format(l1)
      else:
        s1text = "{:.0f} R\n{:.0f}".format(1/k1,l1)
      if abs(k2) < 1/300:
        s2text = "\n{:.0f}".format(l2)
      else:
        s2text = "{:.0f} R\n{:.0f}".format(1/k2,l2)
      cids.append(canvas.create_text([h1[0],h1[1]],
                                     text=s1text,
                                     tags="segment"))
      cids.append(canvas.create_text([h2[0],h2[1]],
                                     text=s2text,
                                     tags="segment"))
    return cids
  def drawExtra(self,canvas,cids=None,**kw):
    if not cids: cids = []
    if self.type is SegType.Straight:
      return cids
    t1 = bezier2_dr(1,self.seg.h1)
    t2 = bezier2_dr(0,self.seg.h2)
    J = self.seg.joint()

    cids.append(canvas.create_line([self.ps.point[0],self.ps.point[1],self.pe.point[0],self.pe.point[1]],tags="segment"))
    cids.append(canvas.create_line([J[0],J[1],J[0]+t1[0],J[1]+t1[1]],tags="segment"))

    cas = []
    nn = 32
    cj,kj,aj,lj = circleparam(self.seg.JC)
    for j in range(nn):
      t = lerp(j/(nn-1),-1,1)
      tt = bc_arclen_t(abs(t),aj if t > 0 else 2*m.pi - aj)
      p = self.seg.evalJ(tt if t > 0 else -tt)
      cas.append(p)
    cids.append(canvas.create_line([(x[0],x[1]) for x in cas],tags="segment"))
    return cids
###########################################################################
class ControlCurve:
  def __init__(self):
    self.isOpen    = True
    self.point     = []
    self.segment   = []
    self.rail      = []
  def getHeightList(self):
    hl = []
    for i,s in enumerate(self.segment):
      if s.heights_need_update:
        s.calc_heights()
        print("segment",i,"needs height update!")
        sys.stdout.flush()
      for hgts in s.heights:
        hl.extend(hgts)
    hl.append(self.segment[0].heights[0][0])
    return hl
  def setHeightList(self,hl):
    """heightlist should contain heights for all segments in right order"""
    i = 0
    for s in self.segment:
      for j,hgts in enumerate(s.heights):
        s.heights[j] = hl[i:i+len(hgts)]
        i += len(hgts)
  def getBankList(self): # for elevation editor
    banklist = []
    tl = 0
    td = 0
    for i,s in enumerate(self.segment):
      sps = s.shapeparameters()
      for j,sp in enumerate(sps):
        b      = s.banking[j]
        hgt    = s.heights[j]
        divnum = len(hgt)
        ep,l,div,k,center = sp
        bl = {'banking': b.angle, 'vlen': l, 'vpos':tl,
                         'divNum': divnum, 'unk': td }
        print (bl)
        sys.stdout.flush()
        banklist.append(bl)
        
        tl += l
        td += divnum
    return banklist
  def length(self):
    "total length of control curve"
    tl = 0
    for s in self.segment:
      tl += s.seg.length()
    return tl
  def lengthsAt(self,seg):
    "length of control curve up to segment and length of segment"
    tl = 0
    for s in self.segment:
      if s is seg:
        return tl, s.length()
      tl += s.seg.length()
    return tl,0
  def segmentAndTAt(self,l):
    seg = None
    tl = l
    sl = 0
    while not seg:
      for s in self.segment:
        sl = s.seg.length()
        if tl > sl:
          tl -= sl
        else:
          seg = s
          break
    t = tl/sl
    return seg,t
  def pointAt(self,l):
    s,t = self.segmentAndTAt(l)
    #print("pointAt",l,s,t,s.seg.eval(t))
    return s.seg.eval(t)
  def tangentAt(self,l):
    s,t = self.segmentAndTAt(l)
    return s.seg.eval_t(t)
  def pointAndTangentAt(self,l):
    s,t = self.segmentAndTAt(l)
    return s.seg.eval(t),s.seg.eval_t(t)
  def nearest(self,p):
    mind = 100000000
    minl = None
    tl = 0
    for s in self.segment:
      d,l = s.seg.dist_sqr_at_l(p)
      if d is not None and mind > d:
        mind = d
        minl = tl + l
        #print("shorter!",d,l)
      tl += s.seg.length()
    return minl

    # internal functions
  def __segmentsof(self,cp):
    return [s for s in self.segment if s.ps is cp or s.pe is cp]
  def __neighborsof(self,seg):
    return [s for s in self.segment if seg.ps is s.pe or seg.pe is s.ps]

  def __setupCp(self,cp):
    so = self.__segmentsof(cp)
    for s in so:
      if s.type is SegType.Straight:
        return self.__setupSeg(s)
      else:
        s.setup()
    return so
  def __setupSeg(self,seg):
    seg.setup()
    nb = self.__neighborsof(seg)
    for s in nb:
      s.setup()
    nb.append(seg)
    return nb
  def fixedSegmentPoints(self,seg):
    # return all points that have to move to leave seg fixed
    cps = set()
    cps.add(seg.ps)
    cps.add(seg.pe)
    for s in self.__neighborsof(seg):
      if s.type is SegType.Straight:
        cps.add(s.ps)
        cps.add(s.pe)
    return cps
  # manipulation methods:
  def setTangent(self,cp,t):
    cp.tangent = t
    return self.__setupCp(cp)

  def setPointAndTangent(self,cp,p,t):
    cp.point   = p
    cp.tangent = t
    return self.__setupCp(cp)

  def transformPoints(self,cps,xform):
    segments = []
    for cp in cps:
      cp.point   = transform_point(cp.point,xform)
      cp.tangent = la.unit(transform_vector(cp.tangent,xform))
      so = self.__segmentsof(cp)
      for s in so:
        if s.type is SegType.Straight:
          so.extend(self.__neighborsof(s))
      segments.extend(so)
    transformable = set()
    reset = set()
    for s in segments:
      if s in reset:
        reset.remove(s)
        transformable.add(s)
      else:
        reset.add(s)
    for s in transformable:
      s.transform(xform)
    for s in reset:
      self.__setupSeg(s)
    return reset.union(transformable)

  def movePoint(self,cp,vec):
    cp.point = cp.point + vec
    return self.__setupCp(cp)

  def moveJoint(self,seg,p):
    J = seg.seg.joint()
    p0 = seg.ps.point
    t0 = seg.ps.tangent
    p1 = seg.pe.point
    t1 = seg.pe.tangent
    tj = biarc_joint_circle_tangent(p0, t0, p1, t1)
    pc,t,a = point_on_circle(p0,p1,tj,p)
    seg.biarc_r = t
    seg.setup()
    d = pc - J # displacement of joint point

    #print (pc,seg.seg.joint(),pc2,pc2-pc)

    return [seg],d[0],d[1]
  def tangentUpdateable(self,cp):
    for s in self.__segmentsof(cp):
      #print("     ",s.type,s.ps.point,s.pe.point)
      if s.type is SegType.Straight:
        return False
    return True

  # all manipulating functions return the affected segments
  def toggleOpen(self,*args):
    if self.isOpen:
      close = CCSegment(self.point[-1],self.point[0],SegType.Biarc,*args)
      self.segment.append(close)
      self.isOpen = False
      return close
    else:
      rem = self.segment.pop()
      self.isOpen = True
      return rem
  # returns the new point, the new segments and all affected segments
  def changeType(self,seg):
    if seg is self.segment[0]: # first segment has to stay straight
      return []
    if (seg.type is SegType.Biarc):
      for s in self.__neighborsof(seg):
        if s.type == SegType.Straight:
          return []
      seg.type = SegType.Straight
    else:
      seg.type = SegType.Biarc
    return self.__setupSeg(seg)
  def insertPoint(self,seg,p,*args):
    cp = CCPoint(p)
    if not self.point: # first point
      self.point.append(cp)

      return cp,None,None
    if not self.segment and len(self.point) == 1: # first segment
      args = [SegType.Straight if x == SegType.Biarc else x for x in args]
      newseg = CCSegment(self.point[0],cp,*args)
      self.point.append(cp)
      self.segment.append(newseg)
      return cp,newseg,newseg

    si = None
    if seg is None: # append at end
      si = len(self.segment)
      newseg = CCSegment(self.point[si],cp,*args)
    else: # split an existing segment
      pe,seg.pe = seg.pe,cp
      seg.setup()
      newseg = CCSegment(cp,pe,*args)
      si = self.segment.index(seg)

    self.point.insert(si+2, cp)
    self.segment.insert(si+1, newseg)

    aff = self.__setupSeg(newseg)

    return cp,newseg,aff
  def appendPoint(self,p,*args):
    return self.insertPoint(None,p,*args)
  def removePoint(self,cp):
    if cp is self.point[0] or cp is self.point[1]:
      return [] # don't remove first point or segment
    
    segs = self.__segmentsof(cp)

    if len(segs) > 1:
      segs[1].ps = segs[0].ps

    self.point.remove(cp)
    self.segment.remove(segs[0])

    aff = []
    if len(segs) > 1:
      aff = self.__setupSeg(segs[1])

    # add removed segs[0] to clear canvas items in redraw
    return [segs[0]] + aff
  def reverse(self):
    #reverse lists and reset start to old point 1 and segment 0
    self.point.reverse()
    self.point.insert(0,self.point.pop())
    self.point.insert(0,self.point.pop())

    self.segment.reverse()
    self.segment.insert(0,self.segment.pop())

    # reverse tangents
    for p in self.point:
      p.tangent = -p.tangent
      #print("point",p.point,p.tangent)

    # reverse start and end points in segments
    for s in self.segment:
      s.reverse()
      #print("segment",s.type,s.ps.point,s.pe.point)

    #print(self.pointAt(0))

    return self.segment # all are affected

  def drawSegment(self,seg,canvas,inclText=True,**config):
    # returns a list of the cids of the canvas items or
    # None if segment isn't in the curve
    if seg in self.segment:
      cids = seg.draw(canvas,**config)
      if inclText:
        seg.drawText(canvas,cids,**config)
        #seg.drawExtra(canvas,cids,**config)
      return cids
    else:
      return None
  def draw(self,canvas,inclText=True,**config):
    # returns a map of segments to a list of cids of the canvas items
    seg2cids = {}
    for s in self.segment:
      seg2cids[s] = s.draw(canvas,**config)
      if inclText:
        s.drawText(canvas,seg2cids[s],**config)
        #s.drawExtra(canvas,seg2cids[s],**config)
    return seg2cids

###########################################################################


def rgb2hex(rgb_tuple):
    """ convert an (R, G, B) tuple to #RRGGBB """
    return '#%02x%02x%02x' % tuple(int(x*255) for x in rgb_tuple)

def hex2rgb(hexstring):
    """ convert #RRGGBB to an (R, G, B) tuple """
    hexstring = hexstring.strip()
    if hexstring[0] == '#': hexstring = hexstring[1:]
    if len(hexstring) != 6:
      raise ValueError("input #%s is not in #RRGGBB format" % hexstring)
    r, g, b = hexstring[:2], hexstring[2:4], hexstring[4:]
    r, g, b = [int(n, 16) for n in (r, g, b)]
    return (r/255, g/255, b/255)


def rgb_lighten_saturate(rgb, amount):
  hls = colorsys.rgb_to_hls(*rgb)
  hls = (hls[0], hls[1] + dl, hls[2] + ds)
  return colorsys.hls_to_rgb(*hls)
def hex_lighten_saturate(hexstr, dl,ds):
  hls = colorsys.rgb_to_hls(*hex2rgb(hexstr))
  hls = (hls[0], hls[1] + dl, hls[2] + ds)
  return rgb2hex(colorsys.hls_to_rgb(*hls))

def style_active(style,lighten,saturate,thicken):
  if "fill" in style and style["fill"]:
    style["activefill"] = hex_lighten_saturate(style["fill"],lighten,saturate)
  if "outline" in style:
    style["activeoutline"] = hex_lighten_saturate(style["outline"],lighten,saturate)
  if "width" in style:
    style["activewidth"] = style["width"] + thicken
  return style
def style_modified(style,lighten,saturate,thicken):
  style = deepcopy(style)
  if "fill" in style and style["fill"]:
    style["fill"] = hex_lighten_saturate(style["fill"],lighten,saturate)
  if "activefill" in style and style["activefill"]:
    style["activefill"] = hex_lighten_saturate(style["activefill"],lighten,saturate)
  if "outline" in style:
    style["outline"] = hex_lighten_saturate(style["outline"],lighten,saturate)
  if "activeoutline" in style:
    style["activeoutline"] = hex_lighten_saturate(style["activeoutline"],lighten,saturate)
  if "width" in style:
    style["width"] = style["width"] + thicken
  if "activewidth" in style:
    style["activewidth"] = style["activewidth"] + thicken
  return style
def style_config_mod(cfg,lighten,saturate,thicken):
  style = {}
  if "fill" in cfg and cfg["fill"][4]:
    style["fill"] = hex_lighten_saturate(cfg["fill"][4],lighten,saturate)
  if "outline" in cfg and cfg["outline"][3] is not cfg["outline"][4]:
    style["outline"] = hex_lighten_saturate(cfg["outline"][4],lighten,saturate)
  if "width" in cfg and cfg["width"][3] is not cfg["width"][4]:
    style["width"] = float(cfg["width"][4]) + thicken
  return style

###########################################################################
# helper class for text popups
class PopupEntry(tk.Toplevel):
  def __init__(self, master, pos, desc, cb, title=None):
    super().__init__(master)
    self.cb = cb
    
    self.transient(master)
    if title:
      self.title(title)
    self.geometry("+%d+%d" % (master.winfo_rootx()+pos[0],
                                master.winfo_rooty()+pos[1]))
    self.l = tk.Label(self, text = desc)
    self.l.pack()
    self.e = tk.Entry(self)
    self.e.pack()
    self.initial_focus = self.e
    self.b = tk.Button(self, text = 'Ok', command = self.cleanup)
    self.b.pack()

    self.e.bind("<Return>",self.cleanup)
    self.e.focus_set()
  def cleanup(self,ev=None):
    value = self.e.get()
    self.destroy()
    self.cb(value)
class PopupAbout(tk.Toplevel):
  def __init__(self, master, pos, desc, title=None):
    super().__init__(master)
    self.transient(master)
    if title:
      self.title(title)
    self.geometry("+%d+%d" % (master.winfo_rootx()+pos[0],
                                master.winfo_rooty()+pos[1]))
    self.l = tk.Label(self, text = desc)
    self.l.pack()
    self.b = tk.Button(self, text = 'Ok', command = self.cleanup)
    self.b.pack()
  def cleanup(self,ev=None):
    self.destroy()

###########################################################################
# unfinished:
class RailItem:
  def __init__(self, start, length, uuid, railtype):
    self.start     = start
    self.length    = length
    self.uuid      = uuid
    self.railtype  = railtype
class Road(RailItem):
  def __init__(self,l,type):
    super().__init__(l,type)
class Decoration(RailItem):
  def __init__(self,l,type):
    super().__init__(l,type)
###########################################################################
class RailManip(FSM):
  def __init__(self,cc,canvas):
    self.cc = cc # TODO: what if new curve is exported???!!!
    self.canvas = canvas
    self.history = []
    self.future  = []

    self.movetag    = "rail"

    s  = Enum("States","Idle Insert Move ChangeType RRem")
    tt = {
      # (State, tag or None, event (None is any event)) -> (State, callback)
      (s.Idle,   None,  "<ButtonPress-1>")   : (s.Insert, self.onRailInsertStart),
      (s.Insert, None,  "<B1-Motion>")       : (s.Insert, self.onRailInsertUpdate),
      (s.Insert, None,  "<ButtonRelease-1>") : (s.Idle,   self.onRailInsertEnd),

      (s.Idle,  "rail",  "<ButtonPress-1>")   : (s.Move,  self.onRailMoveStart),
      (s.Move,  "rail",  "<B1-Motion>")       : (s.Move,  self.onRailMoveUpdate),
      (s.Move,  "rail",  "<ButtonRelease-1>") : (s.Idle,  self.onRailMoveEnd),

      (s.Idle,  "rail",  "<Double-Button-1>") : (s.RRem,  self.onRailRemove),
      (s.RRem,  None,    "<ButtonRelease-1>") : (s.Idle,  None),


      (s.Idle,   "rail",  "<ButtonPress-3>")   : (s.Idle, self.onChangeType),

    }
    super().__init__(s.Idle, tt, self.canvas)

    self.movestyle = {
      "width"           : 0,
      "outline"         : "#9370DB",
      "fill"            : "#9370DB",
      "tags"             : "rail"
    }


    #style_active(self.segstyle,  -0.1, 0,   2)
    style_active(self.movestyle, -0.1, 0.1, 3)
    #style_active(self.rotstyle,  -0.1, 0.1, 3)

    self.selstyle = style_modified(self.movestyle,-0.2,0.2,2)

    self.r_cidmap = {}
    self.imap = {}

    class EvInfo:
      pass
    self.info = EvInfo
    self.info.item = None

    self.type = tk.StringVar()

    self.cr = None

  def addMoveHandle(self, railitem):
    c,t = self.cc.pointAndTangentAt(railitem.l)
    pt = la.perp2ccw(t)
    r = 12
    tp = c + pt*17.5+t*20
    pts = [c, c+pt*20-t*2.5, c+pt*20+t*15, c+pt*17.5+t*17.5, c+pt*15+t*15, c+pt*15+t*2.5]
    poly = [(p[0],p[1]) for p in pts]
    cids = [self.canvas.create_polygon(poly,**self.movestyle),
            self.canvas.create_text([tp[0],tp[1]], text=railitem.type, tags=self.movetag)]
    return cids
  def removeHandles(self):
    self.canvas.delete(self.movetag)
  def addHandles(self):
    self.removeHandles()
    self.cp_cidmap = {}
    #for r in self.cr.rails:
    #  cids = self.addMoveHandle(r)
    #  for cid in cids:
    #    self.r_cidmap[cid] = r
    #  self.imap[r] = cids

  def findClosest(self,ev):
    cx,cy = self.canvas.canvasxy(ev.x, ev.y)
    item = self.canvas.find_closest(cx,cy)[0]
    railitem = self.r_cidmap[item]
    return cx,cy,item,railitem

  def onRailRemove(self,ev):
    #self.historySave()
    cx,cy,item,ri = self.findClosest(ev)
    #print("RailRemove")
    #self.cr.rails.remove(ri)
    self.addHandles()

  def onChangeType(self,ev):
    cx,cy,item,ri = self.findClosest(ev)
    self.info.ri = ri
    self.type.set(ri.type)
    #self.info.type, menu = self.cr.typePopup(canvas)
    menu = tk.Menu(self.canvas, tearoff=0, takefocus=0)

    menu.add_radiobutton(label="Normal", value="NORMAL", variable = self.type, command=self.onChangeTypeEnd)
    menu.add_radiobutton(label="Runoff Right", value="R_RUNOFF_RIGHT", variable = self.type, command=self.onChangeTypeEnd)
    menu.add_radiobutton(label="Runoff Left", value="R_RUNOFF_LEFT", variable = self.type, command=self.onChangeTypeEnd)

    menu.tk_popup(ev.x_root, ev.y_root, entry=0)

  def onChangeTypeEnd(self):
    #print("new type",self.type.get())
    self.info.ri.type =  self.type.get()
    self.addHandles()


  def onRailMoveStart(self,ev):
    cx,cy,item,ri = self.findClosest(ev)
    self.info.ri   = ri
    self.info.item = item

    pass
  def onRailMoveUpdate(self,ev):
    cx,cy = self.canvas.canvasxy(ev.x, ev.y)
    l = self.cc.nearest(la.coords(cx,cy))
    ol = self.info.ri.length()
    self.info.ri.setLength(l)
    op,ot = self.cc.pointAndTangentAt(ol)
    p,t = self.cc.pointAndTangentAt(l)

    a1 = m.atan2(ot[0],ot[1])
    a2 = m.atan2(t[0],t[1])

    angle = a1 - a2

    xform = la.identity()
    xform = la.mul(xform,la.translate(p[0],p[1]))
    xform = la.mul(xform,la.rotate(angle,0,0,1))
    xform = la.mul(xform,la.translate(-op[0],-op[1]))

    cids = self.imap[self.info.ri]
    self.canvas.apply_xform(cids,xform)
    #self.canvas.move(self.info.item,p[0] - op[0],p[1] - op[1])
    pass
  def onRailMoveEnd(self,ev):
    pass
  def onRailInsertStart(self,ev):
    cx,cy = self.canvas.canvasxy(ev.x, ev.y)
    self.info.p = la.coords(cx,cy)
    self.info.l = self.cc.nearest(la.coords(cx,cy))
    self.info.c  = self.cc.pointAt(self.info.l)
    pass
  def onRailInsertUpdate(self,ev):
    cx,cy = self.canvas.canvasxy(ev.x, ev.y)
    self.info.p = la.coords(cx,cy)
    self.info.l = self.cc.nearest(la.coords(cx,cy))
    self.info.c  = self.cc.pointAt(self.info.l)

    if self.info.item:
      self.canvas.delete(self.info.item)
      self.info.item = None

    c = self.info.c
    p = self.info.p
    r = 12
    #bb = (c[0]-r, c[1]-r, c[0]+r, c[1]+r)
    bb = [c[0],c[1],p[0],p[1]]
    self.info.item = self.canvas.create_line(bb,fill="black")
  def onRailInsertEnd(self,ev):
    if self.info.item:
      self.canvas.delete(self.info.item)
      self.info.item = None
    #self.cr.addRoad(self.info.l,"NEW_STUFF")
    self.addHandles()
  def historySave(self):
    self.history.append(deepcopy(self.cr)) # save copy of control curve
    self.future = [] # clear redos
    print("Save", len(self.future),len(self.history))
  def onUndoStart(self,ev):
    self.onUndo(ev) # do the undo
    print("UndoStart", len(self.future),len(self.history))
  def restore(self):
    self.addHandles()
  def onUndo(self,ev):
    if self.history:
      current = self.history.pop()
      self.future.append(deepcopy(self.cr))
      self.cr = current
      self.restore()

    print("Undo", len(self.future),len(self.history))
  def onRedo(self,ev):
    if self.future:
      current = self.future.pop()
      self.history.append(deepcopy(self.cr))
      self.cr = current
      self.restore()

    print("Redo", len(self.future),len(self.history))
###########################################################################
class CCManip(FSM):
  def __init__(self,cc,canvas):
    self.cc = cc
    self.canvas = canvas
    self.history = []
    self.future  = []

    self.segtag     = "segment"
    self.movetag    = "move"
    self.rottag     = "rot"
    self.jointtag   = "joint"

    s  = Enum("States","Idle Select Insert Append Move Rot SelRot SelScale Limbo")
    tt = {
      # (State, tag or None, event (None is any event)) -> (State, callback)

      # move control points, selection or segments
      (s.Idle,   "move", "<ButtonPress-1>")      : (s.Move,   self.onMoveStart),
      (s.Move,   "move", "<B1-Motion>")          : (s.Move,   self.onMoveUpdate),
      (s.Move,   "move", "<ButtonRelease-1>")    : (s.Idle,   self.onMoveEnd),

      (s.Idle,   "joint", "<ButtonPress-1>")      : (s.Move,   self.onJointMoveStart),
      (s.Move,   "joint", "<B1-Motion>")          : (s.Move,   self.onJointMoveUpdate),
      (s.Move,   "joint", "<ButtonRelease-1>")    : (s.Idle,   self.onJointMoveEnd),

      (s.Idle,   "segment", "<ButtonPress-1>")   : (s.Move,   self.onMoveStart),
      (s.Move,   "segment", "<B1-Motion>")       : (s.Move,   self.onMoveUpdate),
      (s.Move,   "segment", "<ButtonRelease-1>") : (s.Idle,   self.onMoveEnd),

      # rotate control point tangents
      (s.Idle,   "rot",  "<ButtonPress-1>")      : (s.Rot,    self.onRotStart),
      (s.Rot,    "rot",  "<B1-Motion>")          : (s.Rot,    self.onRotUpdate),
      (s.Rot,    "rot",  "<ButtonRelease-1>")    : (s.Idle,   self.onRotEnd),

      # append point at end
      (s.Idle,   None,  "<ButtonPress-1>")       : (s.Limbo, None),
      (s.Limbo,  None,  "<B1-Motion>")           : (s.Append, self.onPointAppendStart),
      (s.Append, None,  "<B1-Motion>")           : (s.Append, self.onPointAppendUpdate),
      (s.Append, None,  "<ButtonRelease-1>")     : (s.Idle,   self.onPointAppendEnd),

      # change type of segment
      (s.Idle,   "segment", "<Button-2>")        : (s.Idle,   self.onSegmentChangeType),

      # insert point in segment
      (s.Idle,   "segment", "<Double-Button-1>") : (s.Limbo,   self.onSegmentInsert),

      # remove point from segment
      (s.Idle,   "move", "<Double-Button-1>")    : (s.Limbo,   self.onPointRemove),

      # extra state to consume button press and button release of double click
      (s.Limbo,   None,  "<ButtonRelease-1>")    : (s.Idle, None),


      # selection bindings
      (s.Idle  , None,  "<ButtonPress-3>")       : (s.Select, self.onSelectionStart),
      (s.Select, None,  "<B3-Motion>")           : (s.Select, self.onSelectionUpdate),
      (s.Select, None,  "<ButtonRelease-3>")     : (s.Idle,   self.onSelectionEnd),
      
      (s.Idle,   "move","<Control-Button-3>")    : (s.Select, self.onSelectionToggle),
      (s.Select, "move","<ButtonRelease-3>")     : (s.Idle,  None),
      
      # alternative selection mode for non-3-button mouses
      (s.Idle,   None,  "<Shift-Key-s>")         : (s.Select, None),
      (s.Select, None,  "<ButtonPress-1>")       : (s.Select, self.onSelectionStart),
      (s.Select, None,  "<B1-Motion>")           : (s.Select, self.onSelectionUpdate),
      (s.Select, None,  "<ButtonRelease-1>")     : (s.Select, self.onSelectionEnd),
      (s.Select, "move","<Control-Button-1>")    : (s.Select, self.onSelectionToggle),
      (s.Select, None,  "<Shift-Key-s>")         : (s.Idle,   None),

      # select all
      (s.Idle,   None,   "<Control-Key-a>")      : (s.Idle,   self.onSelectAll),

      # selection rotation and scale bindings
      (s.Idle,   None,  "<Shift-ButtonPress-1>") : (s.SelRot,  self.onSelRotStart),
      (s.SelRot, None,  "<Shift-B1-Motion>")     : (s.SelRot,  self.onSelRotUpdate),
      (s.SelRot, None,  "<ButtonRelease-1>")     : (s.Idle,    self.onSelRotEnd),

      (s.Idle,     None,  "<Control-Button-1>")  : (s.SelScale, self.onSelScaleStart),
      (s.SelScale, None,  "<Control-B1-Motion>") : (s.SelScale, self.onSelScaleUpdate),
      (s.SelScale, None,  "<ButtonRelease-1>")   : (s.Idle,     self.onSelRotEnd),

      (s.Idle,   None,  "<Key-l>")               : (s.Idle, self.onScaleLengthPopup),

      # undo bindings
      (s.Idle,   None,   "<Control-Key-z>")      : (s.Idle,   self.onUndo),
      (s.Idle,   None,   "<Control-Key-y>")      : (s.Idle,   self.onRedo),
     
      # toggle open bindings
      (s.Idle,   None,   "<Key-o>")              : (s.Idle,   self.onToggleOpen),
      # reverse track
      (s.Idle,   None,   "<Key-r>")              : (s.Idle,   self.onReverse),
    }
    super().__init__(s.Idle, tt, self.canvas)

    class EvInfo:
      pass

    self.info = EvInfo
    self.info.selstart = None
    self.info.mod = False

    self.cp_cidmap  = {}
    self.seg_cidmap = {}
    self.imap       = {}
    self.joint_cidmap = {}
    self.jointimap  = {}
    self.selection  = set()

    self.segstyle = {
      "width"           : 8,
      "outline"         : "#BEBEBE",
      "fill"            : "",
      "tags"            : self.segtag
    }
    self.movestyle = {
      "width"           : 8,
      "outline"         : "#B0C4DE",
      "fill"            : "",
      "tags"            : self.movetag
    }
    self.rotstyle = {
      "width"           : 8,
      "outline"         : "#EEDD82",
      "fill"            : "",
      "tags"            : self.rottag
    }
    self.jointstyle = {
      "width"           : 1,
      "outline"         : "#EE82DD",
      "fill"            : "#EE82DD",
      "tags"            : self.jointtag
    }

    style_active(self.segstyle,  -0.1, 0,   2)
    style_active(self.movestyle, -0.1, 0.1, 3)
    style_active(self.rotstyle,  -0.1, 0.1, 3)
    style_active(self.jointstyle,  -0.1, 0.1, 24)

    self.selstyle = style_modified(self.movestyle,-0.2,0.2,2)

    self.lengthdisplay = self.canvas.create_text(self.canvas.canvasxy(10,10),
                                                 text="Length: {:.2f}m".format(self.cc.length()),
                                                 anchor = tk.NW,
                                                 tags="fixed-to-window")


    self.redrawSegments()
    self.addHandles()

  def redrawSegments(self,affected=None,except_seg=[]):
    if affected is None:
      for seg,cids in self.jointimap.items():
        for cid in cids:
          self.canvas.delete(cid)
      # TODO: remove nonexisting segments from imap
      self.canvas.delete(self.segtag)

      seg2cid = self.cc.draw(self.canvas,**self.segstyle)
      self.seg_cidmap = {}
      for s,cids in seg2cid.items():
        self.imap[s] = cids
        for cid in cids:
          self.seg_cidmap[cid] = s
    else:
      for i,a in enumerate(affected):
        if a in self.imap:
          cids = self.imap[a]
          for cid in cids:
            self.canvas.delete(cid)
        ncids = self.cc.drawSegment(a, self.canvas, **self.segstyle)
        if ncids is None:
          self.imap.pop(a)
          for cid in cids:
            self.seg_cidmap.pop(cid)
        else:
          self.imap[a] = ncids
          for cid in ncids:
            self.seg_cidmap[cid] = a
    self.addJointHandles(except_seg)
    self.canvas.tag_raise(self.segtag,"contour")
    try:
      self.canvas.tag_raise(self.segtag,"image")
    except:
      pass

    self.canvas.itemconfigure(self.lengthdisplay,
                              text="Length: {:.2f}m".format(self.cc.length()))
  def addJointHandles(self, except_seg=None):
    if except_seg is None: except_seg = []
    # redraw all joint handles except for segments in except_seg
    for seg,cids in self.jointimap.items():
      if seg in except_seg:
        continue
      for cid in cids:
        self.canvas.delete(cid) # remove joint handles
    for seg in self.cc.segment:
      if seg in except_seg:
        continue
      if seg.type is SegType.Biarc:
        cids = [self.addJointHandle(seg)]
        for cid in cids:
          self.joint_cidmap[cid] = seg
          if seg in self.jointimap:
            self.jointimap[seg].extend(cids)
          else:
            self.jointimap[seg] = cids

  def addRotHandle(self, cp):
    c = cp.point
    t = cp.tangent
    p = c + 35*t
    r = 12
    cid = canvas_create_circle(self.canvas, p, r, **self.rotstyle)
    return cid
  def addMoveHandle(self, cp):
    c = cp.point
    r = 12
    cid = canvas_create_circle(self.canvas, c, r, **self.movestyle)
    return cid
  def addJointHandle(self, seg):
    c = seg.seg.joint()
    r = 3
    cid = canvas_create_circle(self.canvas, c, r, **self.jointstyle)
    return cid
  def removeHandles(self):
    self.canvas.delete(self.movetag)
    self.canvas.delete(self.rottag)
    self.canvas.delete(self.jointtag)
  def addHandles(self):
    self.removeHandles()
    self.cp_cidmap = {}
    for cp in self.cc.point:
      cids = []
      cid1 = self.addMoveHandle(cp)
      self.cp_cidmap[cid1] = cp
      cids.append(cid1)
      if (self.cc.tangentUpdateable(cp)):
        cid2 = self.addRotHandle(cp)
        self.cp_cidmap[cid2] = cp
        cids.append(cid2)
      self.imap[cp] = cids
    self.addJointHandles()
  def findClosest(self,ev):
    cx,cy = self.canvas.canvasxy(ev.x, ev.y)
    item = self.canvas.find_closest(cx,cy)[0]
    return cx,cy,item

  def onSegmentChangeType(self,ev):
    self.historySave()

    cx,cy,item = self.findClosest(ev)

    seg = self.seg_cidmap[ item ]
    aff = self.cc.changeType(seg)
    self.redrawSegments(aff)
    self.addHandles()
  def onReverse(self,ev):
    self.historySave()
    aff = self.cc.reverse()
    self.redrawSegments(aff)
    self.addHandles()

  def onPointAppendStart(self,ev):
    if not self.cc.isOpen: return
    cx,cy = self.canvas.canvasxy(ev.x, ev.y)
    self.info.prev = la.coords(cx,cy)
    self.historySave()
    self.info.cp,self.info.seg,self.info.aff = self.cc.appendPoint(la.coords(cx,cy),SegType.Biarc)
    self.cc.setTangent(self.info.cp,None)
    self.redrawSegments([self.info.seg])
  def onPointAppendUpdate(self,ev):
    if not self.cc.isOpen:
      return
    cx,cy = self.canvas.canvasxy(ev.x, ev.y)
    dx,dy = cx - self.info.prev[0], cy - self.info.prev[1]
    self.info.prev = (cx,cy)
    self.cc.movePoint(self.info.cp,la.coords(dx,dy))
    self.cc.setTangent(self.info.cp,None)
    self.redrawSegments([self.info.seg])
  def onPointAppendEnd(self,ev):
    if not self.cc.isOpen: return
    self.info.prev = None
    self.info.seg = None
    self.info.aff = None
    self.info.cp = None
    self.addHandles()
  def onSegmentInsert(self,ev):
    self.historySave()

    cx, cy, self.info.item = self.findClosest(ev)
    self.info.prev = la.coords(cx,cy)

    seg = self.seg_cidmap[self.info.item]

    cp,seg2,aff = self.cc.insertPoint(seg,la.coords(cx,cy), SegType.Biarc)
    self.redrawSegments(aff)
    self.addHandles()

    # cid1 = self.addMoveHandle(cp)
    # cid2 = self.addRotHandle(cp)

    # self.cp_cidmap[cid1] = cp
    # self.cp_cidmap[cid2] = cp
    # self.imap[cp] = [cid1,cid2]


    # self.canvas.tag_raise(cid1)
    # self.canvas.tag_raise(cid2)

    #self.info.item = cid1
    #self.canvas.focus(cid1)
  def onPointRemove(self,ev):
    self.historySave()
    cx,cy,item = self.findClosest(ev)
    cp = self.cp_cidmap[item]
    aff = self.cc.removePoint(cp)
    self.redrawSegments(aff)
    self.addHandles()
  def onSelectionStart(self,ev):
    self.addHandles()
    self.selection.clear()
    cx,cy = self.canvas.canvasxy(ev.x, ev.y)
    self.info.selstart = la.coords(cx,cy)
    self.info.selcid = None
  def redrawSelection(self):
    if self.info.selstart is not None:
      selpoly = [self.info.selstart[0],self.info.selstart[1],
                 self.info.selstart[0],self.info.selend[1],
                 self.info.selend[0],self.info.selend[1],
                 self.info.selend[0],self.info.selstart[1]]
      if self.info.selcid:
        self.canvas.delete(self.info.selcid)

      self.info.selcid = self.canvas.create_polygon(selpoly,fill="",outline="grey")

    for cid in self.canvas.find_withtag(self.movetag):
      self.canvas.itemconfig(cid, self.movestyle)

    for cp in self.selection:
      cids = self.imap[cp]
      for cid in cids:
        if self.rottag in self.canvas.gettags(cid): # only modify the move handles
          continue
        self.canvas.itemconfig(cid,self.selstyle)

  def onSelectionUpdate(self,ev):
    cx,cy = self.canvas.canvasxy(ev.x, ev.y)
    self.info.selend = la.coords(cx,cy)
    cids = self.canvas.find_overlapping(self.info.selstart[0],self.info.selstart[1],self.info.selend[0],self.info.selend[1])
    self.selection.clear()
    for cid in cids:
      if cid not in self.cp_cidmap:
        continue
      cp = self.cp_cidmap[cid]
      self.selection.add(cp)

    self.redrawSelection()
  def onSelectionEnd(self,ev):
    self.canvas.delete(self.info.selcid)
    self.info.selstart = None
    self.info.selend = None
    self.info.selcid = None
  def onSelectionToggle(self,ev):
    cx,cy,cid = self.findClosest(ev)

    cp = self.cp_cidmap[cid]
    if cp in self.selection:
      self.selection.remove(cp)
    else:
      self.selection.add(cp)

    self.redrawSelection()

  def onSelectAll(self,ev):
    if len(self.selection) == len(self.cc.point):
      self.selection.clear()
    else:
      self.selection = set(self.cc.point)
    self.redrawSelection()
  def onScaleLengthPopup(self,ev):
    #print("onScaleLengthPopup")
    self.pe = PopupEntry(self.canvas.master,(ev.x,ev.y),"Enter new Length:",self.onScaleLengthPopupDone)
    #sys.stdout.flush()
  def onScaleLengthPopupDone(self,length):
    #print("onScaleLengthPopupDone",length)
    desired_length = float(length)
    self.historySave()
    scale = desired_length/self.cc.length()
    bbox = self.canvas.bbox("segment")
    mx = abs(bbox[0]-bbox[2])
    my = abs(bbox[1]-bbox[3])
    scale_origin = (mx,my) # point that stays fixed while scaling
    xform = la.identity()
    xform = la.mul(xform,la.translate(scale_origin[0],scale_origin[1]))
    xform = la.mul(xform,la.scale(scale,scale,scale))
    xform = la.mul(xform,la.translate(-scale_origin[0],-scale_origin[1]))
    self.applySelXForm(xform)
  def onMoveStart(self,ev):
    cx,cy,self.info.item = self.findClosest(ev)
    self.info.prev = la.coords(cx,cy)
    self.info.sel = self.selection
    self.info.seg = None
    if self.info.item in self.cp_cidmap:
      cp = self.cp_cidmap[self.info.item]
      if not self.selection or cp not in self.info.sel:
        self.info.sel = [cp]
    elif self.info.item in self.seg_cidmap:
      seg = self.seg_cidmap[self.info.item]
      self.info.sel = self.cc.fixedSegmentPoints(seg)
      self.info.seg = seg
  def onMoveUpdate(self,ev):
    if not self.info.mod:
      self.historySave()
    self.info.mod = True

    cx,cy = self.canvas.canvasxy(ev.x, ev.y)
    dx,dy = cx - self.info.prev[0], cy - self.info.prev[1]
    self.info.prev = la.coords(cx,cy)

    xform = la.translate(dx,dy)
    total_aff = self.cc.transformPoints(self.info.sel,xform)

    #total_aff = set()
    for cp in self.info.sel:
      #aff = self.cc.movePoint(cp,la.coords(dx,dy))
      cids = self.imap[cp]
      for cid in cids:
        self.canvas.move(cid,dx,dy)
      #for a in aff:
      #  total_aff.add(a)

    if self.info.seg:
      total_aff.remove(self.info.seg)
      cids = self.imap[self.info.seg]
      for cid in cids:
        self.canvas.move(cid,dx,dy)

    self.redrawSegments(total_aff)
  def onMoveEnd(self,ev):
    self.info.item = None
    self.info.prev = None
    self.info.sel = None
    self.info.mod = False
    self.info.seg = None
  def onJointMoveStart(self,ev):
    cx,cy,self.info.item = self.findClosest(ev)
    self.info.seg = self.joint_cidmap[self.info.item]
    self.info.mod = False
  def onJointMoveUpdate(self,ev):
    if not self.info.mod:
      self.historySave()
    self.info.mod = True
    cx,cy = self.canvas.canvasxy(ev.x, ev.y)

    aff,dx,dy = self.cc.moveJoint(self.info.seg,la.coords(cx,cy))
    cids = self.jointimap[self.info.seg]
    for cid in cids:
      self.canvas.move(cid,dx,dy)
    self.redrawSegments(aff,aff)
  def onJointMoveEnd(self,ev):
    self.info.item = None
    self.info.seg  = None
    self.info.prev = None
    self.info.mod  = False
  def onRotStart(self,ev):
    cx,cy = self.canvas.canvasxy(ev.x, ev.y)
    self.info.item = self.canvas.find_closest(cx,cy)[0]
    self.info.prev = la.coords(cx,cy)
  def onRotUpdate(self,ev):
    if not self.info.mod:
      self.historySave()
    self.info.mod = True
    cx,cy = self.canvas.canvasxy(ev.x, ev.y)

    cp = self.cp_cidmap[self.info.item]
    p  = cp.point
    ot = cp.tangent
    t,l  = la.unit_length(la.coords(cx,cy) - p)

    d = 35*t - 35*ot
    da = 70*t - 70*ot

    aff = self.cc.setTangent(cp,t)

    #for s in self.cc.segment:
    #  if cp is s.ps:
    #    print(1/l)
    #    s.biarc_r = 1/l
    #  if cp is s.pe:
    #    print(l)
    #    s.biarc_r = 1/l

    self.canvas.move(self.info.item, d[0], d[1])

    self.redrawSegments(aff)
  def onRotEnd(self,ev):
    self.info.item = None
    self.info.preva = 0
    self.info.mod = False
  def applySelXForm(self,xform):
    aff = self.cc.transformPoints(self.selection,xform)
    self.addHandles()
    self.redrawSegments(aff)
    self.redrawSelection()
  def applySelXFormOld(self,xform):
    total_aff = set()
    for cp in self.selection:
      rcp = transform_point(cp.point,xform)
      rct = la.unit(transform_vector(cp.tangent,xform))

      # dp = la.coords(rcp[0],rcp[1]) - cp.point
      # dt = -35*cp.tangent + dp + 35*la.coords(rct[0],rct[1])
      # da = -70*cp.tangent + dp + 70*la.coords(rct[0],rct[1])

      aff = self.cc.setPointAndTangent(cp,rcp,rct)
      for a in aff:
        total_aff.add(a)
      # cids = self.imap[cp]
      # self.canvas.apply_xform(cids,xform)
      # for cid in cids:
      #   if self.rottag in self.canvas.gettags(cid):
      #     self.canvas.move(cid, dt[0],dt[1])
      #   else:
      #     self.canvas.move(cid, dp[0],dp[1])
      self.addHandles()

      self.redrawSegments(total_aff)
      self.redrawSelection()
  def onSelRotStart(self,ev):
    cx,cy = self.canvas.canvasxy(ev.x, ev.y)
    self.info.prev  = la.coords(cx,cy)
    self.info.preva = None

    c = la.coords(cx,cy)
    r = 40
    self.info.center = canvas_create_circle(self.canvas, c, r)
  def onSelRotUpdate(self,ev):
    if not self.selection:
      return

    cx,cy = self.canvas.canvasxy(ev.x, ev.y)
    dx,dy = cx - self.info.prev[0], cy - self.info.prev[1]

    t,l = la.unit_length(la.coords(dx,dy))
    cura = m.atan2(t[1],t[0])

    if self.info.preva is None:
      self.historySave()
    elif l < 40: # don't rotate too near on center
      pass
    else:
      xform = la.identity()
      xform = la.mul(xform,la.translate(self.info.prev[0],self.info.prev[1]))
      a = cura - self.info.preva
      xform = la.mul(xform,la.rotate(a,0,0,1))
      xform = la.mul(xform,la.translate(-self.info.prev[0],-self.info.prev[1]))
      self.applySelXForm(xform)

    self.info.preva = cura
  def onSelScaleStart(self,ev):
    cx,cy = self.canvas.canvasxy(ev.x, ev.y)
    self.info.prev  = la.coords(cx,cy)
    self.info.preva = None

    c = la.coords(cx,cy)
    r = 60
    self.info.center = canvas_create_circle(self.canvas, c, r)

  def onSelScaleUpdate(self,ev):
    if not self.selection:
      return

    cx,cy = self.canvas.canvasxy(ev.x, ev.y)
    dx,dy = cx - self.info.prev[0], cy - self.info.prev[1]
    t,l = la.unit_length(la.coords(dx,dy))

    cura = l

    if self.info.preva is None:
      if l < 60:
        return
      self.historySave()
    elif l < 10: # don't scale too near on center
      return
    else:
      xform = la.identity()
      xform = la.mul(xform,la.translate(self.info.prev[0],self.info.prev[1]))
      a = cura/self.info.preva
      xform = la.mul(xform,la.scale(a,a,a))
      xform = la.mul(xform,la.translate(-self.info.prev[0],-self.info.prev[1]))
      self.applySelXForm(xform)

    self.info.preva = cura


  def onSelRotEnd(self,ev):
    self.info.prev = None
    self.info.preva = 0
    self.canvas.delete(self.info.center)
    pass

  def historySave(self):
    self.history.append(deepcopy(self.cc)) # save copy of control curve
    self.future = [] # clear redos
    print("Save", len(self.future),len(self.history))
  def onUndoStart(self,ev):
    self.onUndo(ev) # do the undo
    print("UndoStart", len(self.future),len(self.history))
  def restore(self):
    self.selection.clear() # clear selection
    self.redrawSegments()
    self.addHandles()
  def onUndo(self,ev):
    if self.history:
      current = self.history.pop()
      self.future.append(deepcopy(self.cc))
      self.cc = current
      self.restore()

    print("Undo", len(self.future),len(self.history))
  def onRedo(self,ev):
    if self.future:
      current = self.future.pop()
      self.history.append(deepcopy(self.cc))
      self.cc = current
      self.restore()

    print("Redo", len(self.future),len(self.history))
  def onToggleOpen(self,ev,*args):
    self.historySave()
    aff = self.cc.toggleOpen(*args)
    self.redrawSegments([aff])
    self.addHandles()

###########################################################################
class BankingManip(tk.Frame):
  def __init__(self, app, cc, master=None):
    super().__init__(master)
    self.app = app
    self.cc = cc
    self.pack()
    self.setup()
  def setup(self):
    # create a toplevel menu

    self.canvas = CX.CanvasX(self,width=800,height=380)
    self.hbar=tk.Scrollbar(self,orient=tk.HORIZONTAL)
    self.hbar.pack(side=tk.BOTTOM,fill=tk.X)
    self.hbar.config(command=self.canvas.xview)
    self.vbar=tk.Scrollbar(self,orient=tk.VERTICAL)
    self.vbar.pack(side=tk.RIGHT,fill=tk.Y)
    self.vbar.config(command=self.canvas.yview)

    self.canvas.config(scrollregion=(-10,-190,self.cc.length()+10,190),confine=True)
    self.canvas.config(xscrollcommand=self.hbar.set, yscrollcommand=self.vbar.set)

    self.canvas.pack(side=tk.LEFT,expand=True,fill=tk.BOTH)
    self.canvas.focus_set()

    self.imap = {}
    self.textcid = None
    
    self.drawCoords()
    self.drawBanking()
    self.drawTransition(True)

    self.canvas.bind("<Motion>", self.onMouseMotion)
    self.canvas.bind("<MouseWheel>", self.onWheel)
    self.canvas.tag_bind("bank","<B1-Motion>",self.onBankMove)
    self.canvas.tag_bind("transpoint","<B1-Motion>",self.onTransMove)
    self.canvas.bind("<Configure>", self.onConfigure)
  def onConfigure(self, ev):
    self.pack(side=tk.LEFT,expand=True,fill=tk.BOTH)
    self.canvas.pack(side=tk.LEFT,expand=True,fill=tk.BOTH)
  def onWheel(self, ev):
    cx,cy = self.canvas.canvasxy(ev.x,ev.y)
    sf = 1.1
    if (ev.delta < 0): sf = 1/sf
    # scale all objects on canvas
    self.canvas.zoom(cx, cy, sf)
  def onMouseMotion(self,ev):
    l,a = self.canvas.canvasxy(ev.x, ev.y)
    if l < 0: l = 0
    if l > self.cc.length(): l = self.cc.length()
    self.app.drawTrackIndicator(l)
    if self.textcid:
      self.canvas.delete(self.textcid)
    self.textcid = self.canvas.create_text(self.canvas.canvasxy(10,10),
                                           text="{:.2f}m\n{:.4f}°".format(l,a),
                                           anchor = tk.NW, tag="fixed-to-window")

  def drawCoords(self):
    self.canvas.create_line(0,0,self.cc.length(),0, fill="black",tag="grid")
    self.canvas.create_line(0,-180,   0,180, fill="black",tag="grid")
    for i in range(10,180,10):
      self.canvas.create_line(0,+i,self.cc.length(),+i, fill="grey",tag="grid")
      self.canvas.create_line(0,-i,self.cc.length(),-i, fill="grey",tag="grid")
    self.canvas.tag_lower("grid")
  def drawTransition(self,withPoints=False):
    self.canvas.delete("trans")
    if withPoints:
      delitem = []
      for cid,bk in self.imap.items():
        if "transpoint" in self.canvas.gettags(cid):
          delitem.append(cid)
      for cid in delitem:
        self.canvas.delete(cid)
        self.imap.pop(cid)
    tl = 0
    for i,seg in enumerate(self.cc.segment):
      prev = self.cc.segment[i-1]
      next = self.cc.segment[i+1 if i < len(self.cc.segment)-1 else 0]
      shapes_p = prev.shapeparameters()
      shapes = seg.shapeparameters()
      ns = len(shapes)

      bk_p  = (prev.banking[len(shapes_p)-1],seg.banking[0])
      bk_n  = (next.banking[0],seg.banking[ns-1])

      for j,shape in enumerate(shapes):
        ep,l,div,k,center = shape
        bk = seg.banking[j]
        bk1 = bk_p[j]
        bk2 = bk_n[ns-1-j]

        bkm1 = (bk1.angle + bk.angle) / 2
        bkm2 = (bk2.angle + bk.angle) / 2
        self.canvas.create_line(tl, bkm1, tl + bk.prev_len, bk.angle,
                                width = 1, fill = "lightblue", tags = "trans")
        self.canvas.create_line(tl + l - bk.next_len, bk.angle, tl + l, bkm2,
                                width = 1, fill = "lightblue", tags = "trans")
        if withPoints:
          transcid = canvas_create_circle(self.canvas,[tl+bk.prev_len,bk.angle],2,
                                          width = 1, activewidth = 3, fill = "blue", tags = "transpoint")
          self.imap[transcid] = (bk,"prev",tl,tl+l)
          transcid = canvas_create_circle(self.canvas,[tl+l-bk.next_len,bk.angle],2,
                                          width = 1, activewidth = 3, fill = "blue", tags = "transpoint")
          self.imap[transcid] = (bk,"next",tl,tl+l)
        tl += l
  def onTransMove(self,ev):
    l,a = self.canvas.canvasxy(ev.x, ev.y)
    cid = self.canvas.find_withtag("current")[0]
    bk,prev_or_next,bl,el = self.imap[cid]
    dl = 0
    if l < bl:
      l = bl
    if l > el:
      l = el
    if (prev_or_next == "prev"):
      bk.prev_len = l - bl
    else:
      bk.next_len = el - l
    sl,sa,el,ea = self.canvas.coords(cid)
    self.canvas.move(cid,l-(sl+el)/2,0)
    self.drawTransition()
    self.drawBanking()
  def drawBanking(self):
    delitem = []
    for cid,bk in self.imap.items():
      if "bank" in self.canvas.gettags(cid):
        delitem.append(cid)
    for cid in delitem:
      self.canvas.delete(cid)
      self.imap.pop(cid)
    tl = 0
    for i,seg in enumerate(self.cc.segment):
      shapes = seg.shapeparameters()
      for j,shape in enumerate(shapes):
        ep,l,div,k,center = shape
        bk = seg.banking[j]
        bankcid = self.canvas.create_line(tl+bk.prev_len,bk.angle,tl+l-bk.next_len,bk.angle,
                                          width = 3, activewidth = 5,
                                          fill = "blue", tags = "bank")
        self.imap[bankcid] = bk
        tl += l
  def onBankMove(self,ev):
    l,a = self.canvas.canvasxy(ev.x, ev.y)
    cid = self.canvas.find_withtag("current")[0]
    bk = self.imap[cid]
    bk.angle = a
    sl,sa,el,ea = self.canvas.coords(cid)
    self.canvas.move(cid,0,a-sa)
    self.drawTransition(True)
    
###########################################################################
class App(tk.Frame):
  def __init__(self, master=None):
    super().__init__(master)
    self.pack()
    self.setup()
  def askOpenFileName(self):
    path = filedialog.askopenfilename()
    print(path)
    return path
  def askSaveFileName(self):
    path = filedialog.asksaveasfilename()
    print(path)
    return path
  def loadCP(self):
    path = self.askOpenFileName()
    try:
      self.cc = pickle.load(open(path,"rb"))
      self.ccmanip.cc = self.cc
      self.ccmanip.redrawSegments()
      self.ccmanip.addHandles()
      self.recenterTrack()
    except FileNotFoundError:
      print("file not found!")
    #self.ccmanip = CCManip(self.cc,self.canvas)
  def saveCP(self):
    path = self.askSaveFileName()
    try:
      pickle.dump(self.cc,open(path,"wb"))
    except FileNotFoundError:
      print("file not found!")
  def importTed(self):
    path = self.askOpenFileName()
    tedfile = ""
    try:
      with open(path, mode='rb') as file:
        tedfile = file.read()
    except FileNotFoundError:
      print("file not found!")
      return
    self.tedfile = bytearray(tedfile)
    self.hdr     = ted.ted_data_to_tuple("header",ted.header,tedfile,0)
    self.cps     = ted.ted_data_to_tuple_list("cp",ted.cp,tedfile,self.hdr.cp_offset,self.hdr.cp_count)
    self.banks   = ted.ted_data_to_tuple_list("bank",ted.segment,tedfile,self.hdr.bank_offset,self.hdr.bank_count)
    self.heights = ted.ted_data_to_tuple_list("height",ted.height,tedfile,self.hdr.height_offset,self.hdr.height_count)
    self.checkps = ted.ted_data_to_tuple_list("checkpoints",ted.checkpoint,tedfile,self.hdr.checkpoint_offset,self.hdr.checkpoint_count)
    self.road    = ted.ted_data_to_tuple_list("road",ted.road,tedfile,self.hdr.road_offset,self.hdr.road_count)
    self.deco    = ted.ted_data_to_tuple_list("decoration",ted.decoration,tedfile,self.hdr.decoration_offset,self.hdr.decoration_count)

    old_bbox = self.canvas.bbox("segment")

    self.cc = ControlCurve()

    def set_banking(banking,bank):
      banking.angle     = bank.banking
      banking.prev_len  = bank.transition_prev_vlen
      banking.next_len  = bank.transition_next_vlen
      
    def get_segment_heights(heights,bank):
      sh = [heights[j].height for j in range(bank.total_divisions,
                                               bank.total_divisions+bank.divisions)]
      #print(len(sh),bank.divisions)
      #sys.stdout.flush()
      return sh
    width = self.hdr.road_width
    for i,cp in enumerate(self.cps):
      prev = self.cps[i-1]
      lx = (self.cps[i-1].x - cp.x)
      ly = (self.cps[i-1].y - cp.y)
      l = m.sqrt(lx*lx + ly*ly)
      
      #print(i,ted.SegType(cp.segtype),cp.x,cp.y,cp.center_x,cp.center_y, l)
      #sys.stdout.flush()
      
      if (ted.SegType(cp.segtype) == ted.SegType.Straight):
        # straight segment
        p,seg,_ = self.cc.appendPoint(la.coords(cp.x,cp.y),SegType.Straight,width)
        if seg:
          set_banking(seg.banking[0],self.banks[i-1])
          hs = get_segment_heights(self.heights,self.banks[i-1])
          seg.set_heights([hs,[]])
      # second arc of biarc segment
      segarc2  = (ted.SegType(cp.segtype) == ted.SegType.Arc2CCW or
                  ted.SegType(cp.segtype) == ted.SegType.Arc2CW)
      prevsegarc1 = (ted.SegType(prev.segtype) == ted.SegType.Arc1CCW or
                     ted.SegType(cp.segtype) == ted.SegType.Arc1CW)
      segnearly = ted.SegType(cp.segtype) == ted.SegType.NearlyStraight
      prevsegnearly = ted.SegType(prev.segtype) == ted.SegType.NearlyStraight

      if ( segarc2 or segnearly and(prevsegarc1 or prevsegnearly) ):
        p1    = la.coords(self.cps[i-0].x,self.cps[i-0].y)
        joint = la.coords(self.cps[i-1].x,self.cps[i-1].y)
        p0    = la.coords(self.cps[i-2].x,self.cps[i-2].y)

        dx = (cp.x - cp.center_x)
        dy = (cp.y - cp.center_y)

        if ted.SegType(cp.segtype) == ted.SegType.Arc2CCW:
          t = la.unit(la.coords(dy,-dx))
        if ted.SegType(cp.segtype) == ted.SegType.Arc2CW:
          t = la.unit(la.coords(-dy,dx))
        if ted.SegType(cp.segtype) == ted.SegType.NearlyStraight:
          t = la.unit(la.coords(cp.x-prev.x,cp.y-prev.y))

        biarc_r = la.norm(joint-p0)/(la.norm(joint-p0)+la.norm(joint-p1))
        
        #biarc_r = bezier_circle_parameter(p0,p1,self.cc.point[-1].tangent,joint)
        
        p,seg,_ = self.cc.appendPoint(la.coords(cp.x,cp.y),SegType.Biarc,width,biarc_r)
        p.tangent = t

        set_banking(seg.banking[0],self.banks[i-2])
        set_banking(seg.banking[1],self.banks[i-1])
        hs1 = get_segment_heights(self.heights, self.banks[i-2])
        hs2 = get_segment_heights(self.heights, self.banks[i-1])
        seg.set_heights([hs1,hs2])
    # ted file contains first point as last point if track is closed, fix it:
    if self.hdr.is_loop and self.cc.isOpen:
      self.cc.segment[-1].pe = self.cc.segment[0].ps
      self.cc.point.pop() # remove duplicated last point
      self.cc.isOpen = False
    # rebuild
    for s in self.cc.segment: # quick fix for setup update madness while importing
      s.heights_need_update = False
    self.ccmanip.cc = self.cc
    self.ccmanip.redrawSegments()
    self.ccmanip.addHandles()
    self.recenterTrack()
    
  def exportTed(self):
    # convert ted
    TedCp  = ted.ted_data_tuple("cp",ted.cp)
    TedSeg = ted.ted_data_tuple("segment",ted.segment)
    segs = []
    total_l = 0
    total_div = 0
    total_heights = []
    eps = 2**-16

    # control points always include first point, first segment is always straight 
    cps = [TedCp(segtype=int(ted.SegType.Straight.value),
                x=self.cc.segment[0].ps.point[0],
                y=self.cc.segment[0].ps.point[1],
                center_x=0,
                center_y=0)]
    
    for i,seg in enumerate(self.cc.segment):
      if seg.heights_need_update:
        seg.calc_heights()
      for j,sp in enumerate(seg.shapeparameters()):
        ep,l,div,k,center = sp
        bk   = seg.banking[j]
        hgts = seg.heights[j]
        bdiv = len(hgts)
        
        if seg.type is SegType.Straight:
          segtype = ted.SegType.Straight
        elif abs(k) < eps:
          segtype = ted.SegType.NearlyStraight
        elif j == 0:
          segtype = ted.SegType.Arc1CW if k > 0 else ted.SegType.Arc1CCW
        elif j == 1:
          segtype = ted.SegType.Arc2CW if k > 0 else ted.SegType.Arc2CCW
        
        cps.append(TedCp(segtype=int(segtype.value),
                         x=ep[0],
                         y=ep[1],
                         center_x=center[0],
                         center_y=center[1]))
        segs.append(TedSeg(banking=bk.angle,
                           transition_prev_vlen=bk.prev_len,
                           transition_next_vlen=bk.next_len,
                           divisions = bdiv,
                           total_divisions = total_div,
                           vstart = total_l,
                           vlen = l))
        #print("seg",l,div,total_div,l,total_l)
        total_l   += l
        total_div += bdiv
        total_heights.extend(hgts)

    TedHgt = ted.ted_data_tuple("height",ted.height)
    tedheights = []
    for h in total_heights:
      tedheights.append(TedHgt(height=h))
    # always include first height value as last (TODO: check if valid for open tracks)
    tedheights.append(tedheights[0])
    
    eldiff = max(total_heights) - min(total_heights)

    # total_l is the new track length, self.hdr is the header of the imported ted file and contains
    # the old track_length.
    # sf is the linear scale factor for the lengths of road and decoration elements and others
    sf = total_l/self.hdr.track_length

    TedCheck = ted.ted_data_tuple("checkpoint",ted.checkpoint)
    chps = []
    for cp in self.checkps:
      chps.append(TedCheck(vpos3d = cp.vpos3d*sf))

    railroot = raildefs.getRailRoot("rd/andalusia.raildef")
    # rework this function to deliver useful information
    types, uuids, content = raildefs.getRailDict(railroot)

    # convert road railunits
    TedRoad =  ted.ted_data_tuple("road",ted.road)
    roads = [] # holds new road elements, can be more or less than elements in self.road
    for r in self.road: # all road elements in the imported ted file
      # do something useful with above data while converting the roads
      vs = r.vstart3d * sf
      ve = r.vend3d  * sf
      roads.append(TedRoad(uuid = r.uuid, flag=r.flag, vstart3d = vs, vend3d = ve))

    # convert decoration railunits
    TedDeco =  ted.ted_data_tuple("deco",ted.decoration)
    decos = []
    for d in self.deco:
      vl = d.vend3d - d.vstart3d
      vm = (d.vstart3d + d.vend3d) / 2 # scale midpoint
      vms = vm * sf
      vs = vms - l/2
      ve = vms + l/2
      decos.append(TedDeco(uuid=d.uuid,railtype=d.railtype,vstart3d = vs, vend3d = ve,tracktype=d.tracktype))

    # build new tedfile
    hdr = deepcopy(self.hdr)

    tedsz = (ted.ted_data_size(ted.header) +
             ted.ted_data_size(ted.cp) * len(cps) +
             ted.ted_data_size(ted.segment) * len(segs) +
             ted.ted_data_size(ted.height) * len(tedheights) +
             ted.ted_data_size(ted.checkpoint) * len(chps) +
             ted.ted_data_size(ted.road) * len(roads) +
             ted.ted_data_size(ted.decoration) * len(decos) )

    self.tedfile = bytearray(b'\x00'*tedsz)

    hdr = hdr._replace(track_length=total_l,elevation_diff=eldiff)

    o = ted.tuple_to_ted_data(hdr,ted.header,self.tedfile,0)
    hdr = hdr._replace(cp_offset=o,cp_count=len(cps))
    o = ted.tuple_list_to_ted_data(cps,ted.cp,self.tedfile,hdr.cp_offset,hdr.cp_count)
    hdr = hdr._replace(bank_offset=o,bank_count=len(segs))
    o = ted.tuple_list_to_ted_data(segs,ted.segment,self.tedfile,hdr.bank_offset,hdr.bank_count)
    hdr = hdr._replace(height_offset=o,height_count=len(tedheights))
    o = ted.tuple_list_to_ted_data(tedheights,ted.height,self.tedfile,hdr.height_offset,hdr.height_count)
    hdr = hdr._replace(checkpoint_offset=o,checkpoint_count=len(chps))
    o = ted.tuple_list_to_ted_data(chps,ted.checkpoint,self.tedfile,hdr.checkpoint_offset,hdr.checkpoint_count)
    hdr = hdr._replace(road_offset=o,road_count=len(roads))
    o = ted.tuple_list_to_ted_data(roads,ted.road,self.tedfile,hdr.road_offset,hdr.road_count)
    hdr = hdr._replace(decoration_offset=o,decoration_count=len(decos))
    o = ted.tuple_list_to_ted_data(decos,ted.decoration,self.tedfile,hdr.decoration_offset,hdr.decoration_count)


    ted.tuple_to_ted_data(hdr,ted.header,self.tedfile,0)
    return self.tedfile
  def saveTedToFile(self,teddata):
    # write to disk
    path = self.askSaveFileName()
    try:
      with open(path, mode='wb') as file:
        file.write(teddata)
    except FileNotFoundError:
      print("file not found!")
  def exportAndSaveTed(self):
    teddata = self.exportTed()
    self.saveTedToFile(teddata)
  def onAbout(self):
    text="""
    The Ted Editor
    
    by tarnheld with the help of the GTPlanet community
    
    Special Thanks to eran004, MrGrumpy, NingDynasty,
    Outspacer, Pr1vatejoker, Razerman and all i forgot!

    includes the Elevation Editor by eran0004, uploadTed by Razerman
    
    """
    self.about = PopupAbout(self.canvas.master,(10,10),text)
  def onDisclaimer(self):
    text="""
    DISCLAIMER

    By using the Upload TED command, you will possibly violate the
    Playstation Network Terms of use, and that might result in your
    account being banned! While no one so far has been banned by
    uploading track with software other than the official GT6 Track
    Path Editor, there have been bans because of modding cars and time
    trial entries. The Ted Editor including the track upload feature
    is provided for adding awesome new tracks with features not
    possible before to GT6, not for gaining advantage over other
    players, use it wisely."""
    self.disclaimer = PopupAbout(self.canvas.master,(10,10),text)
      
  def uploadTed(self):
    if not self.disclaimer:
      self.onDisclaimer()
    if not self.tedfile:
      self.tedfile = self.exportTed()
    if not self.cookie:
      self.pe = PopupEntry(self.canvas.master,(10,10),"Enter grand-turismo.com session cookie", self.uploadTedWithCookie)
    else:
      self.uploadTedWithCookie(self.cookie)
  def uploadTedWithCookie(self, cookie, title = "track from from ted editor", country="de"):
    username = ut.checkCookieValidity(cookie)
    data     = ut.checkTedDataValidity(self.tedfile)
    self.cookie = cookie # save cookie for later
    if username and data and ut.uploadTedData(data,title,cookie,username):
      print("upload successful!")
    elif not username:
      print("cookie invalid, please log in again")
      self.cookie = None # invalidate cached cookie
    elif not data:
      print("ted file invalid")
    else:
      print("upload failed, maybe more than 30 tracks used already?")
      
      
    # import requests
    # import json
    # headers = {'Cookie': 'JSESSIONID='+cookie}
    # print(headers)
    # r = requests.post('http://www.gran-turismo.com/gb/ajax/get_online_id/', headers=headers)
    # print(r.text)
    # username = json.loads(r.text)["online_id"]
    # if not username:
    #     # Print error message, cookie is invalid
    #     print("Error: Cookie is invalid!")
    #     self.cookie = None
    #     return False
    # else:
    #     files = {"data": ("gt6.ted", self.tedfile)}
    #     data = {'job': (None, '1'), 'user_id': (None, username), 'title': (None, title)}
    #     res = requests.post('https://www.gran-turismo.com/'+country+'/api/gt6/course/', files=files, data=data, headers=headers, verify=False)
    #     uploadResult = json.loads(res.text)["result"]
    #     if uploadResult == 1:
    #         print("Upload succeeded!")
    #         return True
    #     else:
    #         print("Upload failed! Could be due to 30 tracks limit?")
    #         return False
    sys.stdout.flush()
  def recenterTrack(self):
    bbox = self.canvas.bbox("segment")

    ctx,cty = self.canvas.canvasxy(0,0)
    cbx,cby = self.canvas.canvasxy(self.canvas.winfo_width(),self.canvas.winfo_height())
    eox = abs(ctx-cbx)
    eoy = abs(cty-cby)
    enx = abs(bbox[0]-bbox[2])
    eny = abs(bbox[1]-bbox[3])
    aro = eox/eoy
    arn = enx/eny

    if aro < arn:
      sf = eox/enx
    else:
      sf = eoy/eny

    self.canvas.zoom((ctx+cbx)/2,(cty+cby)/2,1/1000)
    self.canvas.zoom((bbox[0]+bbox[2])/2,(bbox[1]+bbox[3])/2,1000*0.9*sf)
  def clearTrackIndicator(self):
    if self.ti_cid:
      self.canvas.delete(self.ti_cid)
      self.ti_cid = None
  def drawTrackIndicator(self,l):
   self.clearTrackIndicator()
   p = self.cc.pointAt(l)
   self.ti_cid = canvas_create_circle(self.canvas,p,10)
  def setup(self):
    # create a toplevel menu
    self.menubar = tk.Menu(self)

    filemenu = tk.Menu(self.menubar, tearoff = 0)
    filemenu.add_command(label="Load Track", command=self.loadCP)
    filemenu.add_command(label="Save Track", command=self.saveCP)
    filemenu.add_command(label="Import TED", command=self.importTed)
    filemenu.add_command(label="Export TED", command=self.exportTed)
    filemenu.add_separator()
    filemenu.add_command(label="Upload TED", command=self.uploadTed)
    filemenu.add_separator()
    filemenu.add_command(label="Import Image", command=self.importImg)
    filemenu.add_command(label="Discard Image", command=self.discardImg)
    filemenu.add_separator()
    filemenu.add_command(label="Quit", command = self.quit)
    self.menubar.add_cascade(label="File", menu = filemenu)
    helpmenu = tk.Menu(self.menubar, tearoff = 0)
    helpmenu.add_command(label="About", command = self.onAbout)
    helpmenu.add_command(label="Upload Disclaimer", command = self.onDisclaimer)
    self.menubar.add_cascade(label="Help", menu = helpmenu)
    
    self.master.config(menu = self.menubar)

    self.canvas = CX.CanvasX(self,width=800,height=600)
    self.hbar=tk.Scrollbar(self,orient=tk.HORIZONTAL)
    self.hbar.pack(side=tk.BOTTOM,fill=tk.X)
    self.hbar.config(command=self.canvas.xview)
    self.vbar=tk.Scrollbar(self,orient=tk.VERTICAL)
    self.vbar.pack(side=tk.RIGHT,fill=tk.Y)
    self.vbar.config(command=self.canvas.yview)
    self.extent = 10000

    self.canvas.config(scrollregion=(-self.extent,-self.extent,self.extent,self.extent),confine=True)
    self.canvas.config(xscrollcommand=self.hbar.set, yscrollcommand=self.vbar.set)

    self.canvas.pack(side=tk.LEFT,expand=True,fill=tk.BOTH)

    self.canvas.bind("<ButtonPress-2>", self.onDragStart)
    self.canvas.bind("<ButtonRelease-2>", self.onDragEnd)
    self.canvas.bind("<Shift-Key-d>", self.onDragToggle)
    self.canvas.bind("<MouseWheel>", self.onWheel)
    self.canvas.bind("<Motion>", self.onDrag)
    self.canvas.bind("<Key-m>", self.onToggleManip)
    self.canvas.bind("<Key-b>", self.onBankingEdit)
    self.canvas.bind("<Key-e>", self.onElevationEdit)

    self.canvas.bind("<Configure>", self.onConfigure)
    self.canvas.focus_set()
    self.drawCoordGrid()
    self.drawContours("hm/andalusia-contours.npz")
    
    self.img = None
    self.simg = None
    self.pimg = None
    self.imgcid = None
    self.bankmanip  = None
    self.bankwindow = None
    self.eleveditor = None
    self.elevwindow = None
    self.ti_cid = None

    self.cookie = None
    self.tedfile = None

    self.disclaimer = None
    self.about = None
    
    self.dragging = False

    self.cc = ControlCurve()

    # initial simple track
    self.cc.appendPoint(la.coords(100,200))
    self.cc.appendPoint(la.coords(100,300))
    self.cc.appendPoint(la.coords(200,450),SegType.Biarc)
    self.cc.appendPoint(la.coords(600,450))
    self.cc.appendPoint(la.coords(700,300),SegType.Biarc)
    self.cc.appendPoint(la.coords(600,200),SegType.Biarc)
    self.cc.appendPoint(la.coords(400,300))
    self.cc.toggleOpen()

    self.ccmanip   = CCManip(self.cc,self.canvas)
    self.railmanip = RailManip(self.cc,self.canvas)
    self.railmanip.stop()
    self.ccmanip.start()
    
  def onBankingEdit(self,ev):
    if not self.bankwindow:
      self.bankwindow = tk.Toplevel(self.master)
      self.bankmanip = BankingManip(self, self.cc, self.bankwindow)
      self.bankwindow.protocol("WM_DELETE_WINDOW",self.onBankingClose)
      self.ccmanip.stop()
      self.ccmanip.removeHandles()
      self.railmanip.stop()
      self.railmanip.removeHandles()
    else:
      self.onBankingClose()
  def onBankingClose(self):
    self.bankmanip = None
    self.bankwindow.destroy()
    self.bankwindow = None
    self.clearTrackIndicator()
    self.ccmanip.addHandles()
    self.ccmanip.start()

  def onElevationEdit(self,ev):
    def make_tokens(structure):
      '''Creates token data (x, y) from an extracted TED structure'''

      def get_bank_index(index, banks):
        bank_index = -1
        for bank in banks:
          if index >= bank['unk']:
            bank_index += 1
          else:
            break
        return bank_index
          
      def get_distance(bank, index):
        vlen = bank['vlen']
        divNum = bank['divNum']
        unk = bank['unk']
        vpos = bank['vpos']
        distance = vlen/divNum*(index-unk)+vpos
        return distance

      heights = structure['mod']
      banks = structure['banks']
            
      index = 0
      for height in heights:
        bank_index = get_bank_index(index, banks)
        bank = banks[bank_index]
        x = get_distance(bank, index)
        structure['tokens'].append((x, height))
        index += 1

      return structure

    heightlist = self.cc.getHeightList()
    banklist = self.cc.getBankList()

    mod = list(heightlist) #creates copy for modding
        
    structure = {'head': '', 'banks': banklist, 'heights': heightlist,
                 'mod': mod, 'tail': '', 'tokens' : []}

    structure = make_tokens(structure)

    if not self.elevwindow:
      self.elevwindow = tk.Toplevel(self.master)
      self.eleveditor = eed.ElevationEditor(self.elevwindow,self)
      self.eleveditor.extract(structure)
      self.elevwindow.protocol("WM_DELETE_WINDOW",self.onElevationClose)
      self.ccmanip.stop()
      self.ccmanip.removeHandles()
      self.railmanip.stop()
      self.railmanip.removeHandles()
    else:
      self.onElevationClose()
  def onElevationClose(self):
    hl = self.eleveditor.get_heightlist()
    self.cc.setHeightList(hl)
    
    self.eleveditor = None
    self.elevwindow.destroy()
    self.elevwindow = None
    self.clearTrackIndicator()
    self.ccmanip.addHandles()
    self.ccmanip.start()
    
      
  def drawContours(self,path):
    import re
    self.contours = np.load("hm/andalusia-contours.npz")
    for a in self.contours.files:
      cs = self.contours[a]
      h = int(re.findall('\d+',a)[0])
      h = h/len(self.contours.files)
      col = colorsys.rgb_to_hsv(0.7,0.9,0.7)
      col = (col[0] - h,col[1],col[2])
      col = colorsys.hsv_to_rgb(*col)
      hexcol = rgb2hex(col)
      for c in cs:
        if len(c):
          cc = [((x[1]-512)/1024*3499.99975586*2,(x[0]-512)/1024*3499.99975586*2) for x in c]
          if la.norm(c[-1] - c[0]) < 0.01:
            self.canvas.create_polygon(cc,fill="",outline=hexcol,width=7,tag="contour")
          else:
            self.canvas.create_line(cc,fill=hexcol,width=7,tag="contour")
    self.canvas.tag_lower("contour")
            

  def onToggleManip(self, ev):
    if self.ccmanip.isStopped():
      self.railmanip.stop()
      self.railmanip.removeHandles()
      self.ccmanip.addHandles()
      self.ccmanip.start()
    else:
      self.ccmanip.stop()
      self.ccmanip.removeHandles()
      self.railmanip.addHandles()
      self.railmanip.start()

  def onDragStart(self, ev):
    self.canvas.focus_set()
    self.canvas.scan_mark(ev.x,ev.y)
    self.dragging = True

  def onDragEnd(self, ev):
    self.dragging = False
  def onDragToggle(self, ev):
    if self.dragging:
      self.dragging = False
    else:
      self.canvas.focus_set()
      self.canvas.scan_mark(self.canvas.winfo_pointerx(),self.canvas.winfo_pointery())
      self.dragging = True

  def onDrag(self,ev):
    #print("Motion",ev.x,ev.y,ev.state)
    sys.stdout.flush()
    if (self.dragging):
      self.canvas.scan_dragto(ev.x,ev.y,1)
      self.adjustImg()

  def onWheel(self, ev):
    #  print("Wheel",ev.delta,ev.state)
    sys.stdout.flush()
    cx,cy = self.canvas.canvasxy(ev.x,ev.y)

    sf = 1.1
    if (ev.delta < 0): sf = 1/sf

    # scale all objects on canvas
    self.canvas.zoom(cx, cy, sf)

    self.adjustImg()
    sys.stdout.flush()

  def onConfigure(self, ev):
    self.pack(side=tk.LEFT,expand=True,fill=tk.BOTH)
    self.canvas.pack(side=tk.LEFT,expand=True,fill=tk.BOTH)
    self.adjustImg()

  def discardImg(self):
    self.img = None
    self.simg = None
    self.pimg = None
    self.canvas.delete(self.imgcid)
    self.imgcid = None
    self.imgbbox= None
  def importImg(self):
    path = self.askOpenFileName()

    try:
      self.img = Image.open(path)
    except FileNotFoundError:
      print("file not found!")
      return

    self.simg = self.img
    self.pimg = ImageTk.PhotoImage(self.img)
    self.imgcid = self.canvas.create_image(0,0, image=self.pimg, anchor = tk.NW, tag="image")
    self.imgbbox = (0,0) + self.img.size
    self.canvas.tag_lower("image","segment")
  def adjustImg(self):
    if self.img:
      xf = self.canvas.xform()
      xf = deepcopy(xf)
      xf = la.inverse(xf)
      
      # canvas coordinate of image origin
      ox,oy = self.canvas.canvasxy(self.imgbbox[0],self.imgbbox[1])
      fx,fy = self.canvas.canvasxy(self.imgbbox[2],self.imgbbox[3])
      cw,ch = self.canvas.winfo_width(),self.canvas.winfo_height() # max image width,height
      bx,by = self.canvas.canvasxy(0,0)
      ex,ey = self.canvas.canvasxy(cw,ch)

      cbx,cby = self.canvas.canvasx(0),self.canvas.canvasy(0)
      cex,cey = self.canvas.canvasx(cw),self.canvas.canvasy(ch)
      cox,coy = self.canvas.canvasx(self.imgbbox[0]),self.canvas.canvasy(self.imgbbox[1])
      cfx,cfy = self.canvas.canvasx(self.imgbbox[2]),self.canvas.canvasy(self.imgbbox[3])

      ix = bx
      iy = by
      iw = cw
      ih = ch

      #print((ox,oy),(fx,fy),(bx,by),(ex,ey))
      #print((ix,iy),(iw,ih))
      
      # scale image contents, max size of cw,ch make sure to not overblow image size
      #self.simg = self.img.transform((iw,ih),Image.AFFINE,data=(xf[0][0],xf[0][1],xf[0][3],xf[1][0],xf[1][1],xf[1][3]))
      self.simg = self.img.transform((iw,ih),Image.AFFINE,data=(xf[0][0],xf[0][1],ix,xf[1][0],xf[1][1],iy))
      self.canvas.coords(self.imgcid,ix,iy) # adjust image origin

      self.pimg = ImageTk.PhotoImage(self.simg)
      self.canvas.itemconfig(self.imgcid, image = self.pimg) # set new image
      self.canvas.tag_lower(self.imgcid,"segment") # just below segments
      #sys.stdout.flush()

  def drawCoordGrid(self):

    self.canvas.create_line(-self.extent,    0,self.extent,   0, fill="grey",tag="grid")
    self.canvas.create_line(    0,-self.extent,   0,self.extent, fill="grey",tag="grid")
    for i in range(1,int(self.extent/100)):
      self.canvas.create_line(-self.extent,  i*100,self.extent, i*100, fill="lightgrey",tag="grid")
      self.canvas.create_line(  i*100,-self.extent, i*100,self.extent, fill="lightgrey",tag="grid")
      self.canvas.create_line(-self.extent, -i*100,self.extent,-i*100, fill="lightgrey",tag="grid")
      self.canvas.create_line( -i*100,-self.extent,-i*100,self.extent, fill="lightgrey",tag="grid")
    self.canvas.tag_lower("grid")



root = tk.Tk()
app = App(root)

app.master.title("The TED Editor")
#app.master.maxsize(1900,1000)

import gc
#gc.set_debug(gc.DEBUG_LEAK)
gc.set_debug(gc.DEBUG_UNCOLLECTABLE)

app.mainloop()
