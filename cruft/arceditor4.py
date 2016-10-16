import sys
from enum import Enum
from copy import deepcopy
import pickle
import itertools as it

import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image

import CanvasX as CX

from linalg import (point,vector,unit,norm,norm2,perp2,para,perp,refl,
                    unit_length,dot,mul,inverse,
                    translate,scale,rotate,identity,
                    hom,proj,vapply,coords)
from fsm import FSM
import read_ted as ted
import math as m
##############################################################################
def lerp(t, a, b):
  return (1-t)*point(a) + t*point(b)

def bezier2(t,ps):
  umt = 1-t
  return umt*umt*ps[0] + 2*umt*t*ps[1] + t*t * ps[2]
  
def circarc_h(t,hs):
  x = bezier2(t,hs)
  return proj(x)

def biarc_h(p0, t0, p1, t1, r):
  t0 = unit(t0)
  t1 = unit(t1)
  chord = point(p0) - point(p1)

  f = dot(t0,t1)
  g = dot(chord,r*t0+t1)
  r = (1/3) * (f+1)/2 + 1 * (1-(f+1)/2)
  r = 1
  c = dot(chord,chord)
  b = 2*dot(chord,r*t0+t1)
  a = 2*r*(dot(t0,t1) - 1)
 
  if a == 0:
    c2 = lerp(0.5,p0,p1)
    c1 = lerp(0.5,p0,c2)
    c3 = lerp(0.5,c2,p1)
    w1 = 1#dot( t0,unit(c2 - p0))
    w2 = 1#dot( t1,unit(p1 - c2))
    return hom(c1,w1),hom(c2,1),hom(c3,w2),1,1
    
  D = b*b - 4*a*c
  if D < 0:
    print (D,"<0")
    beta = norm2(chord) / 4*dot(chord,t0)
  else:
    sqD = D**.5
    beta1 = (-b - sqD) / 2 / a
    beta2 = (-b + sqD) / 2 / a
    
    if beta1 > 0 and beta2 > 0:
        print (beta1,beta2,">0")
        return None,None,None,0,0
    beta = max(beta1, beta2)
  
  if beta < 0:
    print (beta,"<0")
    return None,None,None,0,0

  alpha = beta * r
  ab = alpha + beta 
  c1 = point(p0) + alpha * t0
  c3 = point(p1) - beta  * t1
  c2 = (beta / ab)  * point(c1) + (alpha / ab) * point(c3)

  #print(alpha,beta)

  w1 = dot( t0,unit(c2 - p0))
  w2 = dot( t1,unit(p1 - c2))
  return hom(c1,w1),hom(c2,1),hom(c3,w2),alpha,beta

def biarc_r_from_arcs(a1,a2):
  alpha = norm(a1[0] - proj(a1[1]))
  beta  = norm(a1[3] - proj(a1[4]))
  return alpha/beta

def circparam(p0,p1,p2):
  chord = p2-p0
  n = perp2(unit(p1-p0))
  dotnc = dot(chord,n)

  rad    = norm2(chord) / (2*dotnc)
  center = p0 + n * rad

class Biarc:
  def __init__(self, p0, t0, p1, t1, r):
    self.p0 = p0
    self.p1 = p1
    self.t0 = unit(t0)
    self.t1 = unit(t1)
    self.r = r
    self.bp = biarc_h(self.p0,self.t0,self.p1,self.t1,self.r)
    self.h1 = [hom(self.p0,1),self.bp[0],self.bp[1]]
    self.h2 = [self.bp[1],self.bp[2],hom(self.p1,1)]
  def offset(self,o):
    b = deepcopy(self)

    pt0 = perp2(b.t0)
    pt1 = perp2(b.t1)
    
    c1  = proj(b.bp[0])
    c2  = proj(b.bp[1])
    c3  = proj(b.bp[2])

    t2  = unit(c3-c1)
    pt2 = perp2(t2)

    cc1 = unit(c2 - b.p0)
    cc2 = unit(b.p1 - c2)

    dp1 = dot(b.t0,cc1)
    dp2 = dot(b.t1,cc2)
    
    #t2  = unit(pt0 + refl(pt0,cc1))
    #t3  = unit(pt1 + refl(pt1,cc2))
    t2  = perp2(cc1)
    t3  = perp2(cc2)
    
    b.p0 = b.p0 + o * pt0
    b.p1 = b.p1 + o * pt1

    c2 = c2 + o*pt2
    c1 = c1 + o/dp1*t2
    c3 = c3 + o/dp2*t3
    
    w1 = b.bp[0][2] #dot( b.t0,unit(c2 - b.p0))
    w2 = b.bp[2][2] #dot( b.t1,unit(b.p1 - c2))
    w1 = dot( b.t0,cc1)
    w2 = dot( b.t1,cc2)

    alpha = norm(c1-c2)
    beta  = norm(c3-c2)
    b.r = alpha/beta
    
    b.bp = (hom(c1,w1),hom(c2,1),hom(c3,w2),alpha,beta)
    
    b.h1 = [hom(b.p0,1),b.bp[0],b.bp[1]]
    b.h2 = [b.bp[1],b.bp[2],hom(b.p1,1)]
    
    return b
    
  def eval(self,t):
    if t < 0.5:
      return circarc_h(t*2,self.h1)
    else:
      return circarc_h((t-0.5)*2,self.h2)
  def drawSegment(self,canvas,w,**kw):
    b1 = self.offset(w/2)
    b2 = self.offset(-w/2)
    cas = []
    for j in range(0,50):
      cas.append(b1.eval(j/49))
    for j in range(0,50):
      cas.append(b2.eval(1-j/49))

    return canvas.create_polygon([(x[0],x[1]) for x in cas],**kw)
    
  def draw(self,canvas,**kw):
    #c1,c2,c3,a,b = biarc_h(self.p0,self.t0,self.p1,self.t1,self.r)
    cas = []
    for j in range(0,50):
      cas.append(self.eval(j/49))

    cl = [self.p0,proj(self.bp[0]),proj(self.bp[1]),proj(self.bp[2]),self.p1]
    return canvas.create_line([(x[0],x[1]) for x in cas],**kw)
    #canvas.create_line([(x[0],x[1]) for x in cl],tags="controlcurve",fill="darkgreen")
def group_coords(coordlist, n, fillvalue=0):
  args = [iter(coordlist)] * n
  return it.zip_longest(*args, fillvalue=fillvalue)


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

import colorsys

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
def style_mod(cfg,lighten,saturate,thicken):
  style = {}
  if "fill" in cfg and cfg["fill"][4]:
    style["fill"] = hex_lighten_saturate(cfg["fill"][4],lighten,saturate)
  if "outline" in cfg and cfg["outline"][3] is not cfg["outline"][4]:
    style["outline"] = hex_lighten_saturate(cfg["outline"][4],lighten,saturate)
  if "width" in cfg and cfg["width"][3] is not cfg["width"][4]:
    style["width"] = float(cfg["width"][4]) + thicken
  return style

class CCManip(FSM):
  def __init__(self,cc,canvas):
    self.cc = cc
    self.canvas = canvas
    self.history = []
    self.curhist = 0

    self.segtag  = "segment"
    self.movetag = "move"
    self.rottag  = "rot"

    s  = Enum("States","Idle Select Undo Insert Move Rot SelRot")
    tt = {
      # (State, tag or None, event (None is any event)) -> (State, callback)
      
      # move control points
      (s.Idle,   "move", "<ButtonPress-1>")       : (s.Move,   self.onMoveStart),
      (s.Move,   "move", "<B1-Motion>")           : (s.Move,   self.onMoveUpdate),
      (s.Move,   "move", "<ButtonRelease-1>")     : (s.Idle,   self.onMoveEnd),

      # rotate control point tangents
      (s.Idle,   "rot",  "<ButtonPress-1>")       : (s.Rot,    self.onRotStart),
      (s.Rot,    "rot",  "<B1-Motion>")           : (s.Rot,    self.onRotUpdate),
      (s.Rot,    "rot",  "<ButtonRelease-1>")     : (s.Idle,   self.onRotEnd),

      # change type of segment 
      (s.Idle,   "segment", "<ButtonPress-2>")    : (s.Idle,   self.onSegmentChangeType),

      # insert point in segment
      (s.Idle,   "segment", "<Double-Button-1>")  : (s.Idle,   self.onSegmentInsert),

      # remove point from segment
      (s.Idle,   "move", "<Double-Button-1>")     : (s.Idle,   self.onPointRemove),

      # selection bindings
      (s.Idle,   None,  "<Shift-ButtonPress-1>")  : (s.Select, self.onSelectionStart),
      (s.Select, None,  "<Shift-B1-Motion>")      : (s.Select, self.onSelectionUpdate),
      (s.Select, None,  "<ButtonRelease-1>")      : (s.Idle,   self.onSelectionEnd),

      (s.Idle,   "move", "<Control-Button-1>")    : (s.Idle,   self.onSelectionToggle),

      # selection rotation bindings
      (s.Idle,   None,  "<ButtonPress-1>")       : (s.SelRot,  self.onSelRotStart),
      (s.SelRot, None,  "<B1-Motion>")           : (s.SelRot,  self.onSelRotUpdate),
      (s.SelRot, None,  "<ButtonRelease-1>")     : (s.Idle,    self.onSelRotEnd),

      # undo bindings
      (s.Idle,   None,   "<Control-Key-z>")       : (s.Undo,   self.onUndoStart),
      (s.Undo,   None,   "<Control-Key-z>")       : (s.Undo,   self.onUndo),
      (s.Undo,   None,   "<Control-Key-r>")       : (s.Undo,   self.onRedo),
      (s.Undo,   None,   None)                    : (s.Idle,   None),

      # toggle open bindings
      (s.Idle,   None,   "<Key-o>")               : (s.Idle,   self.onToggleOpen),
    }
    FSM.__init__(self, s.Idle, tt, self.canvas)
    
    class EvInfo:
      pass
    
    self.info = EvInfo

    self.info.selstart = None

    
    self.cp_cidmap  = {}
    self.seg_cidmap = {}
    self.imap       = {}
    self.selection  = set()

    self.segstyle = {
      "width"           : 4,
      "outline"         : "#BEBEBE",
      "fill"            : "",
    }
    self.movestyle = {
      "width"           : 4,
      "outline"         : "#B0C4DE",
      "fill"            : "",
    }
    self.rotstyle = {
      "width"           : 4,
      "outline"         : "#EEDD82",
      "fill"            : "",
    }
    style_active(self.segstyle,  -0.1, 0, 2)
    style_active(self.movestyle, -0.1, 0.1, 3)
    style_active(self.rotstyle,  -0.1, 0.1, 3)

    

    self.createHandles()
    self.redrawSegments()
    
  def redrawSegments(self):
    self.canvas.delete(self.segtag)
    self.seg_cidmap = self.cc.draw(self.canvas,
                                   tag = self.segtag,
                                   **self.segstyle)
    self.canvas.tag_lower(self.segtag)
    
  def addRotHandle(self, cp):
    c = cp.point
    t = cp.tangent
    print (c,t)
    p = c + 35*t
    r = 12
    bb = (p[0]-r, p[1]-r, p[0]+r, p[1]+r)
    cid = self.canvas.create_oval(bb, **self.rotstyle, tags=self.rottag)
    return cid
  def addMoveHandle(self, cp):
    c = cp.point
    r = 12
    bb = (c[0]-r, c[1]-r, c[0]+r, c[1]+r)
    cid = self.canvas.create_oval(bb, **self.movestyle, tags=self.movetag)
    return cid
  def createHandles(self):
    self.canvas.delete(self.movetag)
    self.canvas.delete(self.rottag)
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

  def onSegmentChangeType(self,ev):
    print("onChangeType",ev)

    self.historySave()
    
    cx,cy = self.canvas.canvasxy(ev.x, ev.y)
    item = self.canvas.find_closest(cx,cy)[0]
    
    seg = self.seg_cidmap[ item ]
    self.cc.changeType(seg)
    self.redrawSegments()
    self.createHandles()
    
    sys.stdout.flush()
  def onSegmentInsert(self,ev):
    print("onInsert",ev)

    self.historySave()
    
    cx,cy = self.canvas.canvasxy(ev.x, ev.y)
    self.info.item = self.canvas.find_closest(cx,cy)[0]
    self.info.prev = coords(cx,cy)

    seg = self.seg_cidmap[self.info.item]

    si = self.cc.segmentIndex(seg)
    cp, seg2 = self.cc.insertPoint(si,coords(cx,cy), SegType.Biarc)
    self.redrawSegments()

    cid1 = self.addMoveHandle(cp)
    cid2 = self.addRotHandle(cp)

    self.cp_cidmap[cid1] = cp
    self.cp_cidmap[cid2] = cp
    self.imap[cp] = [cid1,cid2]
    
    self.info.item = cid1
    
    self.canvas.tag_raise(cid1)
    self.canvas.tag_raise(cid2)
    self.canvas.focus(cid1)
    
    sys.stdout.flush()
    pass
  def onPointRemove(self,ev):
    print("onRemove",ev)

    self.historySave()
    
    cx,cy = self.canvas.canvasxy(ev.x, ev.y)
    self.info.item = self.canvas.find_closest(cx,cy)[0]

    cp = self.cp_cidmap[self.info.item]

    self.cc.removePoint(cp)
    self.redrawSegments()
    self.createHandles()
    sys.stdout.flush()
    pass

  def onSelectionStart(self,ev):
    print("onSelStart",ev)
    self.createHandles()
    self.selection = set()
        
    cx,cy = self.canvas.canvasxy(ev.x, ev.y)
    self.info.selstart = coords(cx,cy)
    self.info.selcid = None
    sys.stdout.flush()
    pass

  def redrawSelection(self):
    self.createHandles()

    if self.info.selstart is not None:
      selpoly = [self.info.selstart[0],self.info.selstart[1],
                 self.info.selstart[0],self.info.selend[1],
                 self.info.selend[0],self.info.selend[1],
                 self.info.selend[0],self.info.selstart[1]]
      if self.info.selcid:
        self.canvas.delete(self.info.selcid)
      
      self.info.selcid = self.canvas.create_polygon(selpoly,fill="",outline="grey")

    self.createHandles()
    for cp in self.selection:
      cids = self.imap[cp]
      for cid in cids:
        if self.rottag in self.canvas.gettags(cid):
          continue
        cfg = self.canvas.itemconfig(cid)
        cfg = style_mod(cfg,-0.2,0,2)
        self.canvas.itemconfig(cid,cfg)

  def onSelectionUpdate(self,ev):
    print("onSelUpdate",ev)
    cx,cy = self.canvas.canvasxy(ev.x, ev.y)
    self.info.selend = coords(cx,cy)
    cids = self.canvas.find_overlapping(self.info.selstart[0],self.info.selstart[1],self.info.selend[0],self.info.selend[1])
    for cid in cids:
      if cid not in self.cp_cidmap:
        continue
      cp = self.cp_cidmap[cid]
      self.selection.add(cp)

    self.redrawSelection()

    sys.stdout.flush()
    pass
  def onSelectionToggle(self,ev):
    print("onSelAdd",ev)
    cx,cy = self.canvas.canvasxy(ev.x, ev.y)
    cid = self.canvas.find_closest(cx,cy)[0]
    cp = self.cp_cidmap[cid]
    if cp in self.selection:
      self.selection.remove(cp)
    else:
      self.selection.add(cp)
    
    self.redrawSelection()
    sys.stdout.flush()
    pass
    
  def onSelectionEnd(self,ev):
    print("onSelEnd",ev)
    self.canvas.delete(self.info.selcid)
    self.info.selstart = None
    self.info.selend = None
    self.info.selcid = None
    sys.stdout.flush()
    pass
    
  def onMoveStart(self,ev):
    print("onMoveStart",ev)
    sys.stdout.flush()

    self.historySave()
    
    cx,cy = self.canvas.canvasxy(ev.x, ev.y)
    self.info.item = self.canvas.find_closest(cx,cy)[0]
    self.info.prev = coords(cx,cy)
    
    pass
  def onMoveUpdate(self,ev):
    print("onMoveUpdate",ev.x,ev.y)

    cx,cy = self.canvas.canvasxy(ev.x, ev.y)
    dx,dy = cx - self.info.prev[0], cy - self.info.prev[1]
    self.info.prev = coords(cx,cy)

    sel = self.selection
    cp = self.cp_cidmap[self.info.item]
    if not self.selection or cp not in sel:
      sel = [cp]

    for cp in sel:
      self.cc.movePoint(cp,coords(dx,dy))
      cids = self.imap[cp]
      for cid in cids:
        hc = self.canvas.coords(cid)
        print(hc)
        hct = []
        for c in group_coords(hc,2,0):
          hct.append(c[0]+dx)
          hct.append(c[1]+dy)
          print(hct)
        self.canvas.coords(cid, *hct)


    self.redrawSegments()
    sys.stdout.flush()
    pass
  def onMoveEnd(self,ev):
    print("onMoveEnd",ev)
    sys.stdout.flush()
    
    self.info.item = None
    self.info.prev = None
  def onRotStart(self,ev):
    print("onRotStart",ev)
    sys.stdout.flush()

    self.historySave()

    cx,cy = self.canvas.canvasxy(ev.x, ev.y)
    self.info.item = self.canvas.find_closest(cx,cy)[0]
    self.info.prev = coords(cx,cy)

  def onRotUpdate(self,ev):
    print("onRotUpdate",ev.x,ev.y)

    cx,cy = self.canvas.canvasxy(ev.x, ev.y)
   
    cp = self.cp_cidmap[self.info.item]
    p  = cp.point
    ot = cp.tangent
    t,l  = unit_length(coords(cx,cy) - p)
    
    d = 35*t - 35*ot

    cp.tangent = t

    #for s in self.cc.segment:
    #  if cp is s.ps:
    #    print(1/l)
    #    s.biarc_r = 1/l
    #  if cp is s.pe:
    #    print(l)
    #    s.biarc_r = 1/l
    
    self.canvas.move(self.info.item, d[0], d[1])
    self.redrawSegments()
    
    sys.stdout.flush()
  def onRotEnd(self,ev):
    print("onRotEnd",ev)
    sys.stdout.flush()
    
    self.info.item = None
    self.info.preva = 0

    pass

  def onSelRotStart(self,ev):
    print("onSelRotStart",ev)
    sys.stdout.flush()
    
    self.historySave()

    cx,cy = self.canvas.canvasxy(ev.x, ev.y)
    self.info.prev  = coords(cx,cy)
    self.info.preva = None
    
  def onSelRotUpdate(self,ev):
    print("onSelRotUpdate",ev.x,ev.y)

    if not self.selection:
      return
    
    cx,cy = self.canvas.canvasxy(ev.x, ev.y)
    dx,dy = cx - self.info.prev[0], cy - self.info.prev[1]

    t,l = unit_length(coords(dx,dy))

    cura = m.atan2(t[1],t[0])

    if self.info.preva is None:
      a = 0
    else:
      a = cura - self.info.preva
      if a > 360:
        a -= 360
      
      print("rotate:",a)
      xform = identity()
      xform = mul(xform,translate(self.info.prev[0],self.info.prev[1]))
      xform = mul(xform,rotate(a,0,0,1))
      xform = mul(xform,translate(-self.info.prev[0],-self.info.prev[1]))
      
      for cp in self.selection:
        rcp = vapply(xform,coords(cp.point[0],cp.point[1],0,1))
        rct = vapply(xform,coords(cp.tangent[0],cp.tangent[1],0,0))

        cp.point = coords(rcp[0],rcp[1])
        cp.tangent = coords(rct[0],rct[1])

        self.redrawSegments()
        #self.createHandles()
        self.redrawSelection()
    self.info.preva = cura
    
    sys.stdout.flush()
  def onSelRotEnd(self,ev):
    print("onRotEnd",ev)
    sys.stdout.flush()
    
    self.info.prev = None
    self.info.preva = 0
    pass

  def historySave(self):
    for i in range(self.curhist+1,len(self.history)):
      self.history.pop()
    self.history.append(deepcopy(self.cc))
    self.curhist = len(self.history) - 1
  def onUndoStart(self,ev):
    self.historySave()
    self.onUndo(ev)
  def onUndo(self,ev):
    print("onUndo",self.curhist)
    sys.stdout.flush()
    self.curhist -= 1
    if self.curhist < 0:
      self.curhist = 0
    elif self.curhist < len(self.history):
      self.cc = deepcopy(self.history[self.curhist])
      self.redrawSegments()
      self.createHandles()
  def onRedo(self,ev):
    print("onRedo",self.curhist)
    sys.stdout.flush()
    self.curhist += 1
    if self.curhist >= len(self.history):
      self.curhist = len(self.history) - 1 
    elif self.curhist < len(self.history):
      self.cc = deepcopy(self.history[self.curhist])
      self.redrawSegments()
      self.createHandles()
  def onToggleOpen(self,ev):
    self.cc.toggleOpen()
    self.redrawSegments()
class CCPoint:
  def __init__(self, point = coords(0,0), tangent=None):
    self.point = point
    self.tangent = tangent
  def tangent(self,point):
    self.tangent = unit(point - self.point)
class SegType(Enum):
  Straight = 1
  Biarc = 2
class CCSegment:
  def __init__(self, p1, p2, type = SegType.Straight, width=5, biarc_r=1):
    self.ps = p1
    self.pe = p2
    self.type = type
    self.width = width
    self.biarc_r = biarc_r

  def draw(self,canvas,**kw):
    if self.type is SegType.Straight:
      p0 = self.ps.point
      p1 = self.pe.point
      pt = perp2(self.ps.tangent)
      w = self.width
      poly = [(x[0],x[1]) for x in (p0+pt*w/2,p1+pt*w/2,p1-pt*w/2,p0-pt*w/2)]
      return canvas.create_polygon(poly,**kw)
    else:
      p0 = self.ps.point
      p1 = self.pe.point
      t0 = self.ps.tangent
      t1 = self.pe.tangent
      b  = Biarc(p0, t0, p1, t1, self.biarc_r)
      return b.drawSegment(canvas,self.width,**kw)

class ControlCurve:
  def __init__(self):

    self.isOpen    = True
    self.point     = [ CCPoint() ]
    self.segment   = []

  def movePoint(self,cp,vec):
    cp.point = cp.point + vec
    for s in self.segment:
      if s.type is SegType.Straight:
        if cp is s.ps or cp is s.pe:
          s.ps.tangent = s.pe.tangent = unit(s.pe.point - s.ps.point)
  def tangentUpdateable(self,cp):
    for s in self.segment:
      if s.type is SegType.Straight:
        if cp is s.ps:
          return False
        if cp is s.pe:
          return False
    return True
  def changeType(self,seg):
    si = self.segmentIndex(seg)
    if (seg.type is SegType.Biarc):
      for s in self.segment[si-1:si+1]:
        if s.type == SegType.Straight:
          return
      # adjust tangent
      seg.type = SegType.Straight
      seg.ps.tangent = seg.pe.tangent = unit(seg.pe.point-seg.ps.point)
    else:
      seg.type = SegType.Biarc
    
  def segmentIndex(self,seg):
    try:
      return self.segment.index(seg)
    except ValueError:
      return None
  def pointIndex(self,cp):
    try:
      return self.point.index(cp)
    except ValueError:
      return None
  def toggleOpen(self):
    if self.isOpen:
      close = CCSegment(self.point[-1],self.point[0],SegType.Biarc)
      self.segment.append(close)
      self.isOpen = False
    else:
      self.segment.pop()
      self.isOpen = True
    self.resetTangents()
  def resetTangents(self):
    #print("resetTangents")
    for i,seg in enumerate(self.segment):
      #print("b",seg.ps.point,seg.pe.point,seg.ps.tangent,seg.pe.tangent)
      if seg.type is SegType.Straight:
        seg.ps.tangent = seg.pe.tangent = unit(seg.pe.point - seg.ps.point)
      else:
        prevseg = self.segment[i-1]
        if seg.ps.tangent is None:
          seg.ps.tangent = prevseg.pe.tangent
        if seg.pe.tangent is None:
          seg.pe.tangent = unit(refl(seg.ps.tangent,seg.ps.point - seg.pe.point))
      print(seg.ps.point,seg.pe.point,seg.ps.tangent,seg.pe.tangent)
        
  def insertPoint(self,segment_index,p,*args):
    #print("insertPoint")
    cp = CCPoint(p)
    seg = CCSegment(self.point[segment_index],cp,*args)
    if segment_index in range(len(self.segment)):
      oseg    = self.segment[segment_index]
      if oseg.type is SegType.Straight:
        self.changeType(oseg)
      oseg.ps = cp
      oseg.pe.tangent = None
    self.point.insert(segment_index + 1, cp)
    self.segment.insert(segment_index,seg)
    self.resetTangents()
    return cp,seg
  def appendPoint(self,p,*args):
    return self.insertPoint(len(self.segment),p,*args)
  def removePoint(self,cp):
    #print("removePoint")
    s1,s2 = None,None
    for i,s in enumerate(self.segment):
      if s.pe is cp:
        s1 = s
      elif s.ps is cp:
        s2 = s
    s2.ps = s1.ps
    s2.ps.tangent = None
    s2.pe.tangent = None
    self.point.remove(cp)
    self.segment.remove(s1)
    self.resetTangents()
  def draw(self,canvas,**config):
    cidmap = {}
    for s in self.segment:
      cid = s.draw(canvas,**config)
      cidmap[cid] = s
    return cidmap
    
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
      self.ccmanip = CCManip(self.cc,self.canvas)
    except FileNotFoundError:
      print("file not found!")
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
    hdr     = ted.ted_data_to_tuple("header",ted.header,tedfile,0)
    cps     = ted.ted_data_to_tuple_list("cp",ted.cp,tedfile,hdr.cp_offset,hdr.cp_count)
    banks   = ted.ted_data_to_tuple_list("bank",ted.segment,tedfile,hdr.bank_offset,hdr.bank_count)
    heights = ted.ted_data_to_tuple_list("height",ted.height,tedfile,hdr.height_offset,hdr.height_count)
    checkps = ted.ted_data_to_tuple_list("checkpoints",ted.checkpoint,tedfile,hdr.checkpoint_offset,hdr.checkpoint_count)
    road    = ted.ted_data_to_tuple_list("road",ted.road,tedfile,hdr.road_offset,hdr.road_count)
    deco    = ted.ted_data_to_tuple_list("decoration",ted.decoration,tedfile,hdr.decoration_offset,hdr.decoration_count)

    self.cc = ControlCurve()

    #skip = -1
    for i,cp in enumerate(cps):
      lx = (cps[i-1].x - cp.x)
      ly = (cps[i-1].y - cp.y)
      l = m.sqrt(lx*lx + ly*ly)
      print(i,ted.SegType(cp.segtype),cp.x,cp.y,cp.center_x,cp.center_y, l)
      sys.stdout.flush()
      if (ted.SegType(cp.segtype) == ted.SegType.Straight):
        if i == 0:
          self.cc.movePoint(self.cc.point[0],coords(500+cp.x,cp.y-1500))
        else:
          self.cc.appendPoint(coords(500+cp.x,cp.y-1500),SegType.Straight)
      if (ted.SegType(cp.segtype) == ted.SegType.Arc2CCW or
          ted.SegType(cp.segtype) == ted.SegType.Arc2CW):
        
        self.cc.appendPoint(coords(500+cp.x,cp.y-1500),SegType.Biarc)
        dx = (cp.x - cp.center_x)
        dy = (cp.y - cp.center_y)
        
        r = m.sqrt(dx*dx + dy*dy)
        
        ast = m.atan2(dy, dx);

        if ted.SegType(cp.segtype) == ted.SegType.Arc2CCW:
          t = unit(coords(dy,-dx))
        if ted.SegType(cp.segtype) == ted.SegType.Arc2CW:
          t = unit(coords(-dy,dx))

        
        self.cc.point[-1].tangent = t
        
    self.ccmanip = CCManip(self.cc,self.canvas)
           
    pass
  def exportTed(self):
    pass

  def discardImg(self):
    self.img = None
    self.simg = None
    self.pimg = None
    self.canvas.delete(self.imgcid)
    self.imgcid = None
  def importImg(self):
    path = self.askOpenFileName()
    self.img = Image.open(path)
    self.simg = self.img
    self.pimg = ImageTk.PhotoImage(self.img)
    self.imgcid = self.canvas.create_image(0, 0, image=self.pimg, anchor=tk.NW)

  def setup(self):
    # create a toplevel menu
    self.menubar = tk.Menu(self)
        
    filemenu = tk.Menu(self.menubar, tearoff = 0)
    filemenu.add_command(label="Load Track", command=self.loadCP)
    filemenu.add_command(label="Save Track", command=self.saveCP)
    filemenu.add_command(label="Import TED", command=self.importTed)
    filemenu.add_command(label="Export TED", command=self.exportTed)
    filemenu.add_separator()
    filemenu.add_command(label="Import Image", command=self.importImg)
    filemenu.add_command(label="Discard Image", command=self.discardImg)
    filemenu.add_separator()
    filemenu.add_command(label="Quit", command = self.quit)
    self.menubar.add_cascade(label="File", menu = filemenu)
    self.master.config(menu = self.menubar)

    self.canvas = CX.CanvasX(self,width=800,height=600)
    self.hbar=tk.Scrollbar(self,orient=tk.HORIZONTAL)
    self.hbar.pack(side=tk.BOTTOM,fill=tk.X)
    self.hbar.config(command=self.canvas.xview)
    self.vbar=tk.Scrollbar(self,orient=tk.VERTICAL)
    self.vbar.pack(side=tk.RIGHT,fill=tk.Y)
    self.vbar.config(command=self.canvas.yview)
    self.canvas.config(scrollregion=(-1000,-1000,1000,1000),confine=True)
    self.canvas.config(xscrollcommand=self.hbar.set, yscrollcommand=self.vbar.set)

    self.canvas.pack(side=tk.LEFT,expand=True,fill=tk.BOTH)
    
    self.canvas.bind("<ButtonPress-3>", self.onButton3Press)
    self.canvas.bind("<ButtonRelease-3>", self.onButton3Release)
    self.canvas.bind("<MouseWheel>", self.onWheel)
    self.canvas.bind("<Motion>", self.onMouseMotion)
    
    self.canvas.bind("<Configure>", self.onConfigure)
    self.canvas.focus_set()

    self.img = None #Image.open("pau.png")
    self.simg = None #self.img
    self.pimg = None #ImageTk.PhotoImage(self.img)

    self.imgcid = None#self.canvas.create_image(0, 0, image=self.pimg, anchor=tk.NW)
    #self.canvas.tag_lower(self.imgcid)

    self.dragging = False

    self.drawCoordGrid()

    self.cc = ControlCurve()


    # self.cc.movePoint(self.cc.point[0],coords(100,100))
    # self.cc.appendPoint(coords(200,110))
    # self.cc.appendPoint(coords(520,400),  SegType.Biarc)
    # self.cc.insertPoint(1,coords(320,310),SegType.Biarc)
    # cp,s = self.cc.insertPoint(1,coords(220,210),SegType.Biarc)
    # self.cc.removePoint(cp)

    self.cc.movePoint(self.cc.point[0],coords(100,100))
    self.cc.appendPoint(coords(100,200))
    self.cc.appendPoint(coords(300,200),  SegType.Biarc)
    self.cc.appendPoint(coords(500,200),  SegType.Biarc)
    self.cc.appendPoint(coords(700,200),  SegType.Biarc)

    
    self.cc.toggleOpen()
    self.ccmanip = CCManip(self.cc,self.canvas)

    
  def onConfigure(self, ev):
    self.pack(side=tk.LEFT,expand=True,fill=tk.BOTH)
    self.canvas.pack(side=tk.LEFT,expand=True,fill=tk.BOTH)

  def onButton3Press(self, ev):
    self.canvas.focus_set()
    self.canvas.scan_mark(ev.x,ev.y)
    self.canvas.tag_lower(self.imgcid)
    self.dragging = True
    
  def onButton3Release(self, ev):
    self.dragging = False
    
  def onMouseMotion(self,ev):
    if self.img:
      self.canvas.tag_lower(self.imgcid)
    if (self.dragging):
      self.canvas.scan_dragto(ev.x,ev.y,1)

  def onWheel(self, ev):
    cx,cy = self.canvas.canvasxy(ev.x,ev.y)

    sf = 1.1
    if (ev.delta < 0): sf = 1/sf

    # scale all objects on canvas
    self.canvas.zoom(cx, cy, sf)

    if self.img:
      imgsc = tuple(int(sf * c) for c in  self.simg.size)
      ox,oy = self.canvas.canvasxy(0,0)
      cw,ch = self.canvas.winfo_reqwidth(),self.canvas.winfo_reqheight()
      cow,coh = self.canvas.canvasxy(cw,ch)

      if imgsc[0] > cw and imgsc[1] > ch:
        isz = (cw,ch)
      else:
        isz = imgsc
        
      xf = self.canvas.xform()
      xf = deepcopy(xf)
    
      xf = inverse(xf)
      #self.simg = self.img.transform(self.img.size,Image.AFFINE,data=(xf[0][0],xf[0][1],xf[0][3],xf[1][0],xf[1][1],xf[1][3]))
      self.simg = self.img.transform(imgsc,Image.AFFINE,data=(xf[0][0],0,0,0,xf[1][1],0))
    
      self.pimg = ImageTk.PhotoImage(self.simg)
      #self.canvas.itemconfig(self.imgcid, image = self.pimg)
      x,y = self.canvas.coords(self.imgcid)
      print(x,y,self.img.size,ox,oy,cow,coh)
      if self.imgcid:
        self.canvas.delete(self.imgcid)
      self.imgcid = self.canvas.create_image(0, 0, image=self.pimg, anchor=tk.NW)
      sys.stdout.flush()

    
  def drawCoordGrid(self):
    self.canvas.create_line(-1000,    0,1000,   0, fill="grey",tag="grid")
    self.canvas.create_line(    0,-1000,   0,1000, fill="grey",tag="grid")
    for i in range(1,10):
      self.canvas.create_line(-1000,  i*100,1000, i*100, fill="lightgrey",tag="grid")
      self.canvas.create_line(  i*100,-1000, i*100,1000, fill="lightgrey",tag="grid")
      self.canvas.create_line(-1000, -i*100,1000,-i*100, fill="lightgrey",tag="grid")
      self.canvas.create_line( -i*100,-1000,-i*100,1000, fill="lightgrey",tag="grid")
        
root = tk.Tk()
app = App(root)

app.master.title("The TED Editor")
app.master.maxsize(1900,1000)

app.mainloop()
