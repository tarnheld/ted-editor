import tkinter as tk
import sys

from linalg import point,vector,unit,unit_length,dot,inverse,translate,scale,identity,hom,proj,vapply

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
from enum import Enum

class CpType(Enum):
  Straight  = 1
  Biarc     = 2

class MouseManipulator:
  def __init__(self,manip,app,button,tag,mode):
    self.app = app
    self.button = button
    self.tag = tag
    self.mode = mode
    self.manip = manip

    self._ori = {"cx": 0, "cy": 0, "ox": 0, "oy": 0, "item": None}

    self.tag_bind()
    
  def tag_bind(self):
    if (self.tag):
      self.app.canvas.tag_bind(tag,"<ButtonPress-%d>" % self.button,  self.onButtonPress)
      self.app.canvas.tag_bind(tag,"<ButtonRelease-%d>"% self.button, self.onButtonRelease)
      self.app.canvas.tag_bind(tag,"<B%d-Motion>" % self.button,      self.onButtonMotion)
    else:
      print("<ButtonPress-%d>" % self.button)
      sys.stdout.flush()
      self.app.canvas.bind("<ButtonPress-%d>" % self.button,   self.onButtonPress)
      self.app.canvas.bind("<ButtonRelease-%d>" % self.button, self.onButtonRelease)
      self.app.canvas.bind("<B%d-Motion>" % self.button,       self.onButtonMotion)
    
    pass
  def onButtonPress(self,ev):
    if not self.manip.switchToMode(self.mode): return
    print("%s Press" % (self.mode))
    sys.stdout.flush()
    pass
  def onButtonRelease(self,ev):
    if not self.manip.isInMode(self.mode): return
    print("%s Release" % (self.mode))
    sys.stdout.flush()
    self.manip.leaveMode(self.mode)
    pass
  def onButtonMotion(self,ev):
    if not self.manip.isInMode(self.mode): return
    print("%s Motion" % (self.mode))
    sys.stdout.flush()
    pass

class MoveManipulator(MouseManipulator):
  pass

class Manipulative:
  def __init__(self):
    self._mode = Manip.Idle
  def isInMode(self,mode):
    return self._mode is mode
  def switchToMode(self,mode):
    if self._mode is Manip.Idle:
      self._mode = mode
      return True
    else:
      return False
  def leaveMode(self,mode):
    if self._mode is mode:
      self._mode = Manip.Idle
      return True
    else:
      return False
    
class ControlCurve(Manipulative):
  def __init__(self,app):
    super().__init__()
    self.app = app
    self.cps = []
    self.cls = []
    self.cidmap = {}

    self._ori = {"cx": 0, "cy": 0, "ox": 0, "oy": 0, "item": None}

    self.cps = [point((100,100)),point((100,300)),point((400,300)),point((400,600))]
    self.cls = [0.5,0.5,0.5]
    self.mode = 1

    
    
    self.mode = 0
    self.manip = Manip.Idle

    self.inserter = MouseManipulator(self,self.app,1,None,Manip.Insert)
    
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
    cx,cy = self.app.w2c(event.x, event.y)
    ox,oy = self.app.w2o(event.x, event.y)
    self._ori["item"] = self.app.canvas.find_closest(cx,cy)[0]
    self._ori["cx"] = cx
    self._ori["cy"] = cy
    self._ori["ox"] = ox
    self._ori["oy"] = oy
  def onHandleButtonRelease(self, event):
    if (self.manip is not Manip.Move): return
    print("HandleRelease")
    sys.stdout.flush()
    self._ori["item"] = None
    self._ori["cx"] = 0
    self._ori["cy"] = 0
    self._ori["ox"] = 0
    self._ori["oy"] = 0
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
    
    cx,cy = self.app.w2c(event.x, event.y)
    ox,oy = self.app.w2o(event.x, event.y)
    delta_cx = cx - self._ori["cx"]
    delta_cy = cy - self._ori["cy"]
    delta_ox = ox - self._ori["ox"]
    delta_oy = oy - self._ori["oy"]
    self._ori["cx"] = cx
    self._ori["cy"] = cy
    self._ori["ox"] = ox
    self._ori["oy"] = oy
    
    # move the control point handle
    self.app.canvas.move(self._ori["item"], delta_cx, delta_cy)
    # move the control point in object coordinates
    i = self.cidmap[self._ori["item"]]
    self.cps[i] = point(self.cps[i]) + point((delta_ox,delta_oy))

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
      
      if (c1==None): return
      
      cl = [p0,c1,c2,c3,p1]
      cas = []
      for j in range(0,11):
        cas.append((circarc_h(j/10,[hom(p0,1),c1,c2])))
      for j in range(0,11):
        cas.append((circarc_h(j/10,[c2,c3,hom(p1,1)])))
          
          
      #clc  = [self.app.o2c(*x) for x in cl]
      carc = [self.app.o2c(*x) for x in cas]
          
      #self.app.canvas.create_line(clc,fill="lightblue",tag="obj")
      self.app.canvas.create_line(carc,fill="darkgreen",tag="obj")
    else:
      for i in range(2,len(self.cps)):
        cas = []
        for j in range(0,11):
          cas.append((circarc(j/10,self.cls[i-2],self.cps[i-2:i+1])))
          #print(cas)
        cl   = [self.app.o2c(*x) for x in self.cps[i-2:i+1]]
        carc = [self.app.o2c(*x) for x in cas]
        self.app.canvas.create_line(cl,fill="lightblue",tag="obj")
        self.app.canvas.create_line(carc,fill="darkgreen",tag="obj")
        
    sys.stdout.flush()

    
class App(tk.Frame):
  def __init__(self, master=None):
    super().__init__(master)
    self.pack()
    self.setup()
    
  def setup(self):
    self.canvas = tk.Canvas(self,width=800,height=600)
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

    self.OtoC = identity()
    self.sc  = 1.0
    self.dragging = False

    self.drawCoordGrid()

    self.cc = ControlCurve(self)
    self.cc.addControlHandles()
    self.cc.addControlLevers()
    self.cc.drawControlCurve()

  def w2c(self,wx,wy):
    cx,cy = self.canvas.canvasx(wx), self.canvas.canvasy(wy)
    return cx,cy
  def o2c(self,ox,oy):
    c = vapply(self.OtoC,(ox,oy,0,1))
    return c[0],c[1]
  def c2o(self,cx,cy):
    c2om = inverse(self.OtoC)
    o = vapply(c2om,(cx,cy,0,1))
    return o[0],o[1]
  def w2o(self,wx,wy):
    cx,cy = self.w2c(wx,wy)
    return self.c2o(cx,cy)
  
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
    cx,cy = self.w2c(ev.x,ev.y)
    ox,oy = self.w2o(ev.x,ev.y)
    print(ev.x,ev.y,cx,cy,ox,oy)
    sys.stdout.flush()
    
  def onWheel(self, ev):
    cx,cy = self.w2c(ev.x,ev.y)
    ox,oy = self.w2o(ev.x,ev.y)
    sf = 1.1
    if (ev.delta < 0): sf = 1/sf
    self.sc  *= sf
    
    # scale all objects on canvas
    self.canvas.scale(tk.ALL, cx, cy, sf, sf)
    
    # update object to canvas coordinate transformation
    self.OtoC = mul(self.OtoC,translate(ox,oy))
    self.OtoC = mul(self.OtoC,scale(sf,sf))
    self.OtoC = mul(self.OtoC,translate(-ox,-oy))
    
    # update scrollregion
    sr = self.canvas["scrollregion"]
    self.canvas.config(scrollregion=[float(x)*sf for x in sr.split()],confine=True)

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
