import tkinter as tk
import CanvasX as CX
import sys
from enum import Enum
from copy import deepcopy
from linalg import (point,vector,unit,norm,norm2,perp2,para,perp,refl,
                    unit_length,dot,inverse,
                    translate,scale,identity,
                    hom,proj,vapply,coords,line_intersect)

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
  
  c = dot(chord,chord)
  b = 2*dot(chord,r*t0+t1)
  a = 2*r*(dot(t0,t1) - 1)
 
  if a == 0:
    c2 = lerp(0.5,p0,p1)
    w1 = dot( t0,unit(c2 - p0))
    w2 = dot( t1,unit(p1 - c2))
    return hom(t0,w1),hom(c2,1),hom(-t1,-w2),0,0
    
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

    canvas.create_polygon([(x[0],x[1]) for x in cas],**kw)
    
  def draw(self,canvas,**kw):
    #c1,c2,c3,a,b = biarc_h(self.p0,self.t0,self.p1,self.t1,self.r)
    cas = []
    for j in range(0,50):
      cas.append(self.eval(j/49))

    cl = [self.p0,proj(self.bp[0]),proj(self.bp[1]),proj(self.bp[2]),self.p1]
    canvas.create_line([(x[0],x[1]) for x in cas],**kw)
    #canvas.create_line([(x[0],x[1]) for x in cl],tags="controlcurve",fill="darkgreen")
##############################################################################
class FSM:
  def __init__(self, initialState, transitions, widget):
    self._state  = initialState
    self._tt     = transitions
    self._widget = widget

    events = set()
    # find unique events
    for state, tag, event in self._tt.keys():
      if event is not None: # None is any event
        events.add((tag,event))
    # bind all unique events
    for tag,event in events:
      # event thunk that saves the event info in extra arguments
      def _trans(ev,self = self,te = (tag,event)):
        return self.transition(ev,te[0],te[1])
      if (tag):
        self._widget.tag_bind(tag,event,_trans)
      else:
        self._widget.bind(event,_trans)
  # rebind all events for the_tag
  def update_tag_binds(self,the_tag):
    events = set()
    for state,tag,event in self._tt.keys():
      if (tag is the_tag and event is not None):
        events.add((tag,event))
    for tag,event in events:
      def _trans(ev,self = self,te = (tag,event)):
        return self.transition(ev,te[0],te[1])
      self._widget.tag_unbind(tag,event)
      self._widget.tag_bind(tag,event,_trans)
    
  def transition(self,ev,tag,event):
    #print ("transition from state",self._state,tag,event)
    tr = None
    key = (self._state,tag,event)
    if tk.CURRENT:
      tags = ev.widget.gettags(tk.CURRENT)
      if not tag in tags or not key in self._tt:
        #print ("no tags transition found",key)
        key = (self._state,None,event)
    if not key in self._tt:
      # check for any event transition
      key = (self._state,None,None)
    if key in self._tt:
      print ("transition found",key,tr)
      tr = self._tt[key]
      if tr[1]: # callback
        tr[1](ev) 
        self._state = tr[0] # set new state
      else:
        self._state = tr[0] # set new state
        self.transition(ev,tag,event) # retransition
    else:
      print ("no transition found",key)
    sys.stdout.flush()

class CCManip(FSM):
  def __init__(self,cc,canvas):
    self.cc = cc
    self.canvas = canvas
    self.history = []
    self.curhist = 0
    s  = Enum("States","Idle Undo Insert Move Rot")
    tt = {
      # (State, tag or None, event (None is any event)) -> (State, callback)
      
      (s.Idle,   "move", "<ButtonPress-1>")       : (s.Move,   self.onMoveStart),
      (s.Move,   "move", "<B1-Motion>")           : (s.Move,   self.onMoveUpdate),
      (s.Move,   "move", "<ButtonRelease-1>")     : (s.Idle,   self.onMoveEnd),
      (s.Idle,   "rot",  "<ButtonPress-1>")       : (s.Rot,    self.onRotStart),
      (s.Rot,    "rot",  "<B1-Motion>")           : (s.Rot,    self.onRotUpdate),
      (s.Rot,    "rot",  "<ButtonRelease-1>")     : (s.Idle,   self.onRotEnd),
      (s.Idle,   "segment", "<ButtonPress-2>")    : (s.Idle,   self.onChangeType),
      (s.Idle,   "segment", "<ButtonPress-1>")    : (s.Insert, self.onInsert),
      (s.Insert, "segment", "<B1-Motion>")        : (s.Insert, self.onMoveUpdate),
      (s.Insert, "segment", "<ButtonRelease-1>")  : (s.Idle,   self.onMoveEnd),
      (s.Idle,   None,   "<Control-Key-z>")       : (s.Undo,   self.onUndoStart),
      (s.Undo,   None,   "<Control-Key-z>")       : (s.Undo,   self.onUndo),
      (s.Undo,   None,   "<Control-Key-r>")       : (s.Undo,   self.onRedo),
      (s.Undo,   None,   None)                    : (s.Idle,   None)
    }
    FSM.__init__(self, s.Idle, tt, self.canvas)

    self.info = {"cx": 0, "cy": 0, "item": None}
    self.cidmap = {}
    self.imap   = []
    
    self.addHandles()

  def onChangeType(self,ev):
    print("onChangeType",ev)

    self.historySave()
    
    cx,cy = self.canvas.canvasxy(ev.x, ev.y)
    self.info["item"] = self.canvas.find_closest(cx,cy)[0]
    #self.info["item"] = tk.CURRENT# self.canvas.find_closest(cx,cy)[0]

    i = self.cidmap[self.info["item"]]
    self.cc.changeType(i)
    self.cc.draw(self.canvas)
    self.addHandles()
    self.update_tag_binds("segment")
    sys.stdout.flush()
    pass

  def addRotHandle(self,i):
    c = self.cc.point[i]
    t = self.cc.tangent[i]
    #pt = perp2(t)
    #p1 = c + 10*t + 10*pt
    #p2 = c + 10*t - 10*pt
    #p3 = c + 35*t
    #pl = [(x[0],x[1]) for x in [p1,p2,p3]]
    p = c + 35*t
    r = 12
    bb = (p[0]-r, p[1]-r, p[0]+r, p[1]+r)
    cid = self.canvas.create_oval(bb,
                                  fill="lightgoldenrod",
                                  activefill="goldenrod",
                                  tags="rot")
    return cid
  def addMoveHandle(self,i):
    c = self.cc.point[i]
    r = 12
    bb = (c[0]-r, c[1]-r, c[0]+r, c[1]+r)
    cid = self.canvas.create_oval(bb,
                                  fill="lightsteelblue",
                                  activefill="steelblue",
                                  tags="move")
    return cid
  
  def addHandles(self):
    self.canvas.delete("move")
    self.canvas.delete("rot")
    for i,c in enumerate(self.cc.point):
      cid1 = self.addMoveHandle(i)
      self.cidmap[cid1] = i
      if (i is 0 or self.cc.segtype[i-1] is self.cc.SegType.Straight):
        self.imap.insert(i,[cid1])
      else:  
        cid2 = self.addRotHandle(i)
        self.cidmap[cid2] = i
        self.imap.insert(i,[cid1,cid2])

    #self.update_tag_binds("move")
  
  def onInsert(self,ev):
    print("onInsert",ev)

    self.historySave()
    
    cx,cy = self.canvas.canvasxy(ev.x, ev.y)
    self.info["item"] = self.canvas.find_closest(cx,cy)[0]
    self.info["cx"] = cx
    self.info["cy"] = cy

    i = self.cidmap[self.info["item"]]

    self.cc.insertPoint(i,coords(cx,cy),self.cc.SegType.Biarc,None,30,1)
    self.cc.draw(self.canvas)

    cid1 = self.addMoveHandle(i)
    cid2 = self.addRotHandle(i)
    
    self.imap.insert(i,[cid1,cid2])

    for j,cids in enumerate(self.imap):
      for cid in cids:
        self.cidmap[cid] = j
    
    self.info["item"] = cid1
    
    self.canvas.tag_raise(self.info["item"],"move")
    self.canvas.focus(self.info["item"])
    
    sys.stdout.flush()
    pass
  def onMoveStart(self,ev):
    print("onMoveStart",ev)
    sys.stdout.flush()

    self.historySave()
    
    cx,cy = self.canvas.canvasxy(ev.x, ev.y)
    self.info["item"] = self.canvas.find_closest(cx,cy)[0]
    self.info["cx"] = cx
    self.info["cy"] = cy
    
    pass
  def onMoveUpdate(self,ev):
    print("onMoveUpdate",ev.x,ev.y)

    cx,cy = self.canvas.canvasxy(ev.x, ev.y)
    dx,dy = cx - self.info["cx"],cy - self.info["cy"]
    self.info["cx"],self.info["cy"] = cx,cy
    i = self.cidmap[self.info["item"]]
    cids = self.imap[i]
    for cid in cids:
      self.canvas.move(cid, dx, dy)
    self.cc.movePoint(i,coords(dx,dy))
    self.cc.draw(self.canvas)
    
    print(i,"->",self.info["item"])
    sys.stdout.flush()
    pass
  def onMoveEnd(self,ev):
    print("onMoveEnd",ev)
    sys.stdout.flush()
    
    self.info = {"item" : None, "cx" : 0, "cy" : 0 }

    pass
  def onRotStart(self,ev):
    print("onRotStart",ev)
    sys.stdout.flush()

    self.historySave()

    cx,cy = self.canvas.canvasxy(ev.x, ev.y)
    self.info["item"] = self.canvas.find_closest(cx,cy)[0]
    self.info["cx"] = cx
    self.info["cy"] = cy
    
    pass
  def onRotUpdate(self,ev):
    print("onRotUpdate",ev.x,ev.y)

    cx,cy = self.canvas.canvasxy(ev.x, ev.y)
    i = self.cidmap[self.info["item"]]
    p = self.cc.point[i]
    ot = self.cc.tangent[i]
    t = unit(coords(cx,cy) - p)

    d = 35*t - 35*ot

    self.cc.tangent[i] = t
    
    self.canvas.move(self.info["item"], d[0], d[1])
    self.cc.draw(self.canvas)
    
    sys.stdout.flush()
    pass
  def onRotEnd(self,ev):
    print("onRotEnd",ev)
    sys.stdout.flush()
    
    self.info = {"item" : None, "cx" : 0, "cy" : 0 }

    pass
  def historySave(self):
    for i in range(self.curhist + 1,len(self.history)):
      self.history.pop(i)
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
      self.cc.draw(self.canvas)
      self.addHandles()
  def onRedo(self,ev):
    print("onRedo",self.curhist)
    sys.stdout.flush()
    self.curhist += 1
    if self.curhist >= len(self.history):
      self.curhist = len(self.history) - 1 
    elif self.curhist < len(self.history):
      self.cc = deepcopy(self.history[self.curhist])
      self.cc.draw(self.canvas)
      self.addHandles()

      
class ControlCurve:
  def __init__(self,canvas):

    self.isOpen    = False
    self.SegType   = Enum("SegType", "Straight Biarc End")
    self.point     = [ coords(0,0) ]
    self.tangent   = [None]
    self.biarc_r   = []
    self.segwidth  = []
    self.segtype   = []

    self.cidmap    = {}

    self.color = "gray"
    self.tag = "segment"
    self.movePoint(0,coords(100,100))
    self.appendPoint(coords(200,110),self.SegType.Straight,None,30)
    #self.appendPoint(coords(320,310),self.SegType.Biarc,30,1)
    self.appendPoint(coords(520,400),self.SegType.Biarc,None,30,1)
    self.insertPoint(2,coords(320,310),self.SegType.Biarc,None,30,1)
    self.insertPoint(2,coords(620,610),self.SegType.Straight,None,30,1)
    self.removePoint(2)

  def movePoint(self,i,trans):
    self.point[i] = self.point[i] + trans
  def moveSegment(self,i,trans):
    self.point[i-1] = self.point[i-1] + trans
    self.point[i]   = self.point[i] + trans
  def appendPoint(self,point,type,tangent=None,width=10,r=0):
    self.point.append(point)
    self.tangent.append(tangent)
    self.segtype.append(type)
    self.segwidth.append(width)
    self.biarc_r.append(r)
    return len(self.segtype)
  def changeType(self,i):
    ot = self.segtype[i]
    if (ot is self.SegType.Biarc):
      # adjust tangent
      tp = self.SegType.Straight
      p0 = self.point[i-1]
      p1 = self.point[i]
      nt = unit(p1-p0)
      self.tangent[i]   = nt
    else:
      tp = self.SegType.Biarc
    self.segtype[i] = tp
  def insertPoint(self,i,point,type,tangent=None,width=10,r=0):
    self.point.insert(i,point)
    self.tangent.insert(i,tangent)
    self.segtype.insert(i,type)
    self.segwidth.insert(i,width)
    self.biarc_r.insert(i,r)
    
  def removePoint(self,i):
    self.point.pop(i)
    self.segtype.pop(i)
    self.segwidth.pop(i)
    self.biarc_r.pop(i)
    self.tangent.pop(i)
    
  def drawPoint(self,canvas,i):
    p = self.point[i]
    cid = canvas.create_oval(p[0]-5, p[1]-5, p[0]+5, p[1]+5, 
                             outline=self.color,
                             fill='',
                             tags=self.tag)
    
    
  def drawStraight(self,canvas,i):
    p0 = self.point[i-1]
    p1 = self.point[i]
    self.tangent[i] = unit(p1-p0)

    pt = perp2(self.tangent[i])
    w = self.segwidth[i-1]
    poly = [(x[0],x[1]) for x in (p0+pt*w/2,p1+pt*w/2,p1-pt*w/2,p0-pt*w/2)]
    canvas.create_polygon(poly, fill = "light"+self.color, activefill=self.color, tag = self.tag, outline="grey")
    self.drawPoint(canvas,i-1)
    self.drawPoint(canvas,i)
  def drawBiarc(self,canvas,i):
    p0 = self.point[i-1]
    p1 = self.point[i]
    if self.tangent[i-1] is None:
      t0 = self.point[i-1] - self.point[i-2]
    else:
      t0 = self.tangent[i-1]
    if self.tangent[i] is None:
      t1 = unit(refl(t0,self.point[i-1] - self.point[i]))
      self.tangent[i] = t1
    else:
      t1 = self.tangent[i]

    w = self.segwidth[i-1]
      
    b  = Biarc(p0, t0, p1, t1, self.biarc_r[i-1])
    b.drawSegment(canvas, w, fill = "light"+self.color, activefill=self.color, tag = self.tag, outline="grey")
    canvas.tag_lower(self.tag)

    #self.drawPoint(canvas,i-1)
    #self.drawPoint(canvas,i)
    
  def draw(self,canvas):
    canvas.delete(self.tag)
    for i,t in enumerate(self.segtype):
      if t is self.SegType.Straight:
        self.drawStraight(canvas,i+1)
      if t is self.SegType.Biarc:
        self.drawBiarc(canvas,i+1)
    if not(self.isOpen):
      self.tangent[0] = unit(self.point[1]-self.point[0])
      self.drawBiarc(canvas,0)

class App(tk.Frame):
  def __init__(self, master=None):
    super().__init__(master)
    self.pack()
    self.setup()
    
  def setup(self):
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

    self.dragging = False

    self.drawCoordGrid()

    self.cc = ControlCurve(self.canvas)
    self.cc.draw(self.canvas)
    self.ccmanip = CCManip(self.cc,self.canvas)
  
  def onConfigure(self, ev):
    self.pack(side=tk.LEFT,expand=True,fill=tk.BOTH)
    self.canvas.pack(side=tk.LEFT,expand=True,fill=tk.BOTH)

  def onButton3Press(self, ev):
    self.canvas.focus_set()
    self.canvas.scan_mark(ev.x,ev.y)
    self.dragging = True
    
  def onButton3Release(self, ev):
    self.dragging = False
    
  def onMouseMotion(self,ev):
    if (self.dragging):
      self.canvas.scan_dragto(ev.x,ev.y,1)

  def onWheel(self, ev):
    cx,cy = self.canvas.canvasxy(ev.x,ev.y)

    sf = 1.1
    if (ev.delta < 0): sf = 1/sf
       
    # scale all objects on canvas
    self.canvas.zoom(cx, cy, sf)

  def drawCoordGrid(self):
    self.canvas.create_line(-1000,    0,1000,   0, fill="grey",tag="grid")
    self.canvas.create_line(    0,-1000,   0,1000, fill="grey",tag="grid")
    for i in range(1,10):
      self.canvas.create_line(-1000,  i*100,1000, i*100, fill="lightgrey",tag="grid")
      self.canvas.create_line(  i*100,-1000, i*100,1000, fill="lightgrey",tag="grid")
      self.canvas.create_line(-1000, -i*100,1000,-i*100, fill="lightgrey",tag="grid")
      self.canvas.create_line( -i*100,-1000,-i*100,1000, fill="lightgrey",tag="grid")
        
        
app = App()

app.master.title("The TED Editor")
app.master.maxsize(1900,1000)

app.mainloop()
