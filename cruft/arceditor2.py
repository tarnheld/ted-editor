import tkinter as tk
import sys

from linalg import (point,vector,unit,norm2,norm,
                    unit_length,dot,inverse,
                    translate,scale,identity,
                    hom,proj,vapply,coords,line_intersect,perp2)

##############################################################################
def lerp(t, a, b):
  return (1-t)*point(a) + t*point(b)

def bezier2(t,ps):
  umt = 1-t
  return umt*umt*ps[0] + 2*umt*t*ps[1] + t*t * ps[2]
  
def circarc(t,v,ps):
  n1,l1 = unit_length(vector(ps[1],ps[0]))
  n2,l2 = unit_length(vector(ps[1],ps[2]))
  l = min(l1,l2)


  p0 = hom(point(ps[1]) - l * v * n1,1)
  p2 = hom(point(ps[1]) - l * v * n2,1)

  w = dot(n1,unit((p2[0]-p0[0],p2[1]-p0[1])))
  
  p1 = hom(point(ps[1]),w)
  
  x = bezier2(t,[p0,p1,p2])
  
  return proj(x)

def circarc_h(t,hs):
  x = bezier2(t,hs)
  return proj(x)

def biarc(p0, t0, p1, t1, r):
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
    return None,None,None,0,0
    
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

  
  return c1,c2,c3,alpha,beta

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

  print(alpha,beta)

  w1 = dot( t0,unit(c2 - p0))
  w2 = dot( t1,unit(p1 - c2))
  return hom(c1,w1),hom(c2,1),hom(c3,w2),0,0
  #return c1,c2,c3,alpha,beta

def circparam(p0,p1,p2):
  chord = p2-p0
  n = perp2(unit(p1-p0))
  dotnc = dot(chord,n)

  rad    = norm2(chord) / (2*dotnc)
  center = p0 + n * rad

  return center,rad
def biarc_r_from_arcs(a1,a2):
  alpha = norm(a1[0] - proj(a1[1]))
  beta  = norm(a1[3] - proj(a1[4]))
  return alpha/beta

##############################################################################
from enum import Enum
class Manip(Enum):
  Idle   = 1
  Move   = 2
  Insert = 3
class FSM:
  def __init__(self,initialState,transitions,widget):
    self._state  = initialState
    self._tt     = transitions
    self._widget = widget
    events = set()
    
    for state,tag,event in transitions.keys():
      events.add((tag,event))
    for tag,event in events:
      def _trans(ev,self = self,te = (tag,event)):
        return self.transition(ev,te[0],te[1])
      print ("event",tag,event)
      if (tag):
        self._widget.tag_bind(tag,event,_trans)
      else:
        self._widget.bind(event,_trans)

  def transition(self,ev,tag,event):
    #print ("transition from state",self._state,tag,event)
    tr = None
    if tk.CURRENT:
      tags = ev.widget.gettags(tk.CURRENT)
      key = (self._state,tag,event)
      if not tag in tags or not key in self._tt:
        #print ("no tags transition found",key)
        key = (self._state,None,event)
      if key in self._tt:
        tr = self._tt[key]
        #print ("transition found",tr)
        tr[1](ev) # callback
        self._state = tr[0] # set new state
      #else:
        #print ("no transition found",key)
      #sys.stdout.flush()


class FSMTest(FSM):
  def __init__(self,canvas):
    s  = Enum("States","Idle Insert")
    tt = {
      # (State, tag or None, event)         -> (State, callback)
    
      (s.Idle,   None, "<ButtonPress-1>")   : (s.Insert, self.onInsertStart),
      (s.Insert, None, "<B1-Motion>")       : (s.Insert, self.onInsertUpdate),
      (s.Insert, None, "<ButtonRelease-1>") : (s.Idle,   self.onInsertEnd),
      (s.Idle,   "handle", "<Shift-ButtonPress-1>") : (s.Idle, self.onSelect),
    }
    FSM.__init__(self, s.Idle, tt, canvas)
  def onInsertStart(self,ev):
    print("onInsertStart",ev)
    sys.stdout.flush()
    pass
  def onInsertUpdate(self,ev):
    print("onInsertUpdate",ev.x,ev.y)
    pass
  def onInsertEnd(self,ev):
    print("onInsertEnd",ev)
    pass
  def onSelect(self,ev):
    print("onSelect")
    pass
  
class ControlCurve:
  def __init__(self,app):
    super().__init__()
    self.app = app
    self.cps = []
    self.cls = []
    self.cidmap = {}

    self._ori = {"cx": 0, "cy": 0, "item": None}

    self.cps = [coords(100,100),coords(100,300),coords(400,300),coords(400,600)]
    self.cls = [0.5,0.5,0.5]
    self.mode = 1

    
    
    self.mode = 0
    self.manip = Manip.Idle

    self.inserter = FSMTest(self.app.canvas)
    
  def addControlHandles(self):
    for i,c in enumerate(self.cps):
      cid = self.app.canvas.create_oval(c[0]-15, c[1]-15, c[0]+15, c[1]+15, 
                                        outline="red",
                                        fill="red",
                                        tags="handle")
      self.cidmap[cid] = i
    self.app.canvas.tag_bind("handle","<ButtonPress-1>", self.onHandleButtonPress)
    self.app.canvas.tag_bind("handle","<ButtonRelease-1>", self.onHandleButtonRelease)
    self.app.canvas.tag_bind("handle","<B1-Motion>", self.onHandleMotion)
    #self.app.canvas.bind("<ButtonPress-1>", self.onInsertButtonPress)
    #self.app.canvas.bind("<ButtonRelease-1>", self.onInsertButtonRelease)
    #self.app.canvas.bind("<B1-Motion>", self.onInsertButtonMotion)
    self.app.bind('<Key-m>', self.MKeyPressed)
  def MKeyPressed(self,event):
    print("Mode Switch",self.mode)
    sys.stdout.flush()
    self.mode = 1 - self.mode
    self.drawControlCurve()
  def onInsertButtonPress(self,ev):
    if (self.manip is not Manip.Idle): return
    self.manip = Manip.Insert
    print("InsertionPress")
    sys.stdout.flush()
  def onInsertButtonRelease(self,ev):
    if (self.manip is not Manip.Insert): return
    print("InsertionRelease")
    sys.stdout.flush()
    self.manip = Manip.Idle
  def onInsertButtonMotion(self,ev):
    if (self.manip is not Manip.Insert): return 
    print("InsertionMotion")
    sys.stdout.flush()
    pass
  def addControlLevers(self):
    pass
    # for i in range(1,len(self.cps)):
    #   d,l = unit_length(vector(self.cps[i],self.cps[i-1]))
    #   dh = (-d[1],d[0])
    #   lv = self.cls[i-1]
    #   lp = vadd(self.cps[i-1],vmul(lv*l,d))
    #   lp1 = vadd(lp,vadd(vmul(- 5,d),vmul(-15,dh)))
    #   lp2 = vadd(lp,vadd(vmul(- 5,d),vmul( 15,dh)))
    #   lp3 = vadd(lp,vadd(vmul(  5,d),vmul( 15,dh)))
    #   lp4 = vadd(lp,vadd(vmul(  5,d),vmul(-15,dh)))
    #   lpoly = [lp1,lp2,lp3,lp4]
    #   cid = self.app.canvas.create_polygon(lpoly,
    #                                        outline="blue",
    #                                        fill="blue",
    #                                        tags="lever")
    #   self.cidmap[cid] = i
    #self.app.canvas.tag_bind("lever","<ButtonPress-1>", self.onHandleButtonPress)
    #self.app.canvas.tag_bind("lever","<ButtonRelease-1>", self.onHandleButtonRelease)
    #self.app.canvas.tag_bind("lever","<B1-Motion>", self.onLeverMotion)
  def onHandleButtonPress(self, event):
    if (self.manip is not Manip.Idle): return

    self.manip = Manip.Move
        
    print("HandlePress")
    sys.stdout.flush()
    cx,cy = self.app.canvas.canvasxy(event.x, event.y)
    self._ori["item"] = self.app.canvas.find_closest(cx,cy)[0]
    self._ori["cx"] = cx
    self._ori["cy"] = cy
  def onHandleButtonRelease(self, event):
    if (self.manip is not Manip.Move): return
    print("HandleRelease")
    sys.stdout.flush()
    self._ori["item"] = None
    self._ori["cx"] = 0
    self._ori["cy"] = 0
    self.manip = Manip.Idle
  # def onLeverMotion(self, event):
  #   cx,cy = self.app.w2c(event.x, event.y)
  #   ox,oy = self.app.w2o(event.x, event.y)
  #   delta_cx = cx - self._ori["cx"]
  #   delta_cy = cy - self._ori["cy"]
  #   delta_ox = ox - self._ori["ox"]
  #   delta_oy = oy - self._ori["oy"]
  #   self._ori["cx"] = cx
  #   self._ori["cy"] = cy
  #   self._ori["ox"] = ox
  #   self._ori["oy"] = oy
    
  #   # move the control point handle
  #   self.app.canvas.move(self._ori["item"], delta_cx, delta_cy)
  #   # move the control point in object coordinates
  #   i = self.cidmap[self._ori["item"]]
  #   self.cps[i] = point(self.cps[i]) + point((delta_ox,delta_oy))

  #   self.drawControlCurve()
  def onHandleMotion(self, event):
    if (self.manip is not Manip.Move): return
    print("HandleMotion")
    sys.stdout.flush()
    
    cx,cy = self.app.canvas.canvasxy(event.x, event.y)
    delta_cx = cx - self._ori["cx"]
    delta_cy = cy - self._ori["cy"]
    self._ori["cx"] = cx
    self._ori["cy"] = cy
    
    # move the control point handle
    self.app.canvas.move(self._ori["item"], delta_cx, delta_cy)
    # move the control point in object coordinates
    i = self.cidmap[self._ori["item"]]
    self.cps[i] = self.cps[i] + coords(delta_cx,delta_cy)
    print("delta=",(delta_cx,delta_cy))
    self.drawControlCurve()

  def drawControlCurve(self):
    self.app.canvas.delete("obj")
    if (self.mode):
      p0,p1 = point(self.cps[1]),point(self.cps[2])
      t0 = vector(self.cps[1],self.cps[0])
      t1 = vector(self.cps[3],self.cps[2])

      r = 1
      #for v in range(5):
        #r = 0.2*(1-v/5)+1.8*(v/5)
      c1,c2,c3,a,b = biarc_h(p0,t0,p1,t1,r)
      
      cx,rx = circparam(p0,proj(c1),proj(c2))
      cy,ry = circparam(proj(c2),proj(c3),p1)

      cid = self.app.canvas.create_oval(cx[0]-15, cx[1]-15, cx[0]+15, cx[1]+15, 
                                        outline="black",
                                        fill="green",
                                        tags="obj")
      cid = self.app.canvas.create_oval(cy[0]-15, cy[1]-15, cy[0]+15, cy[1]+15, 
                                        outline="black",
                                        fill="green",
                                        tags="obj")

      
      if (c1==None): return
      
      cl = [p0,c1,c2,c3,p1]
      cas = []
      for j in range(0,11):
        cas.append((circarc_h(j/10,[hom(p0,1),c1,c2])))
      for j in range(0,11):
        cas.append((circarc_h(j/10,[c2,c3,hom(p1,1)])))
          
          
      #clc = [*x for x in cl]
      carc = [(x[0],x[1]) for x in cas]
          
      #self.app.canvas.create_line(clc,fill="lightblue",tag="obj")
      self.app.canvas.create_line(carc,fill="darkgreen",tag="obj")
    else:
      for i in range(2,len(self.cps)):
        cas = []
        for j in range(0,11):
          cas.append((circarc(j/10,self.cls[i-2],self.cps[i-2:i+1])))
          #print(cas)
        cl   = [(x[0],x[1]) for x in self.cps[i-2:i+1]]
        carc = [(x[0],x[1]) for x in cas]
        self.app.canvas.create_line(cl,fill="lightblue",tag="obj")
        self.app.canvas.create_line(carc,fill="darkgreen",tag="obj")
        
    sys.stdout.flush()


import CanvasX as CX
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
    
    self.canvas.bind("<ButtonPress-1>", self.onButton1Press)
    self.canvas.bind("<ButtonPress-3>", self.onButton3Press)
    self.canvas.bind("<ButtonRelease-3>", self.onButton3Release)
    self.canvas.bind("<MouseWheel>", self.onWheel)
    self.canvas.bind("<Motion>", self.onMouseMotion)
    
    self.canvas.bind("<Configure>", self.onConfigure)
    self.focus_set()

    self.dragging = False

    self.drawCoordGrid()

    self.cc = ControlCurve(self)
    self.cc.addControlHandles()
    self.cc.addControlLevers()
    self.cc.drawControlCurve()
  
  def onConfigure(self, ev):
    #print("OnConfigure")
    self.pack(side=tk.LEFT,expand=True,fill=tk.BOTH)
    self.canvas.pack(side=tk.LEFT,expand=True,fill=tk.BOTH)

  def onButton3Press(self, ev):
    self.canvas.scan_mark(ev.x,ev.y)
    self.dragging = True
    
  def onButton3Release(self, ev):
    self.dragging = False
    
  def onMouseMotion(self,ev):
    if (self.dragging):
      self.canvas.scan_dragto(ev.x,ev.y,1)
        
  def onButton1Press(self, ev):
    self.focus_set()
    #cx,cy = self.w2c(ev.x,ev.y)
    #ox,oy = self.w2o(ev.x,ev.y)
    #print(ev.x,ev.y,cx,cy,ox,oy)
    #sys.stdout.flush()
    
  def onWheel(self, ev):
    cx,cy = self.canvas.canvasxy(ev.x,ev.y)

    sf = 1.1
    if (ev.delta < 0): sf = 1/sf
       
    # scale all objects on canvas
    self.canvas.zoom(cx, cy, sf)
    
    # update scrollregion
    #sr = self.canvas["scrollregion"]
    #self.canvas.config(scrollregion=[float(x)*sf for x in sr.split()],confine=True)

  def drawCoordGrid(self):
    self.canvas.create_line(-1000,    0,1000,   0, fill="grey",tag="grid")
    self.canvas.create_line(    0,-1000,   0,1000, fill="grey",tag="grid")
    for i in range(1,10):
      self.canvas.create_line(-1000,  i*100,1000, i*100, fill="lightgrey",tag="grid")
      self.canvas.create_line(  i*100,-1000, i*100,1000, fill="lightgrey",tag="grid")
      self.canvas.create_line(-1000, -i*100,1000,-i*100, fill="lightgrey",tag="grid")
      self.canvas.create_line( -i*100,-1000,-i*100,1000, fill="lightgrey",tag="grid")
        
        
app = App()

app.master.title("The CP Viewer")
app.master.maxsize(1900,1000)

app.mainloop()
