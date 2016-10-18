import tkinter as tk
import linalg as la
import itertools as it
def group_coords(coordlist, n, fillvalue=0):
  args = [iter(coordlist)] * n
  return it.zip_longest(*args, fillvalue=fillvalue)

class CanvasX(tk.Canvas):
    """Canvas widget with zoom support."""
    def __init__(self, master=None, cnf={}, **kw):
        """Construct a canvas widget with the parent MASTER.

        Valid resource names: background, bd, bg, borderwidth, closeenough,
        confine, cursor, height, highlightbackground, highlightcolor,
        highlightthickness, insertbackground, insertborderwidth,
        insertofftime, insertontime, insertwidth, offset, relief,
        scrollregion, selectbackground, selectborderwidth, selectforeground,
        state, takefocus, width, xscrollcommand, xscrollincrement,
        yscrollcommand, yscrollincrement."""
        super().__init__(master, cnf, **kw)
        self._xform = la.identity()

    def _w2c(self,wx,wy):
        cx,cy = super().canvasx(wx), super().canvasy(wy)
        return cx,cy
    def _o2c(self,ox,oy):
        c = la.vapply(self._xform,la.point((ox,oy,0,1)))
        return c[0],c[1]
    def _o2cv(self,ox,oy):
        c = la.vapply(self._xform,la.point((ox,oy,0,0)))
        return c[0],c[1]
    def _c2o(self,cx,cy):
        xformi = la.inverse(self._xform)
        o = la.vapply(xformi,(cx,cy,0,1))
        return o[0],o[1]
    def _w2o(self,wx,wy):
        cx,cy = self._w2c(wx,wy)
        return self._c2o(cx,cy)

    def xform(self):
        return self._xform

    def apply_xform(self,cids,xform):
        # 'arc', 'bitmap', 'image', 'line', 'oval', 'polygon', 'rectangle', 'text', or 'window'
        for cid in cids:
            t = self.type(cid)
            if t == "polygon" or t == "line" or t == "text":
                #print("poly style",t)
                crds  = [la.coords(x,y,0,1) for x,y in group_coords(self.coords(cid),2,0)]
                #print("pre ",crds)
                tcrds = [la.vapply(xform,c) for c in crds]
                #print("post",tcrds)
                tkcrds = [(c[0],c[1]) for c in tcrds]
                self.coords(cid,tkcrds)
            elif t == 'arc' or t == 'oval' or t == 'rectangle':
                #print("arc style",t)
                # transform only center
                crds = self.coords(cid)
                center = la.coords((crds[0]+crds[2])/2,(crds[1]+crds[3])/2,0,1)
                w,h = crds[2]-crds[0],crds[3]-crds[1]
                tc = la.vapply(xform,center)
                #tw = vapply(xform,la.coords(w,0,0,0))
                #th = vapply(xform,la.coords(0,h,0,0))
                self.coords(cid,tc[0]-w/2,tc[1]-h/2,tc[0]+h/2,tc[1]+h/2)
                
    def zoom(self, x, y, s):
        """Zoom Canvas with center x,y and scale s."""

        cx,cy = self._o2c(x,y)
        
        self._xformz = la.translate(cx,cy)
        self._xformz = la.mul(self._xformz,la.scale(s,s))
        self._xformz = la.mul(self._xformz,la.translate(-cx,-cy))
        
        sr = self["scrollregion"]
        if sr:
            srd = [self.tk.getdouble(x) for x in sr.split()]
            c1 = la.vapply(self._xformz,(srd[0],srd[1],0,1))
            c2 = la.vapply(self._xformz,(srd[2],srd[3],0,1))
            self.config(scrollregion=(c1[0],c1[1],c2[0],c2[1]))

        super().scale("fixed-to-window",cx,cy,1/s,1/s)
        self.scale(tk.ALL,x,y,s,s)
        self._xform = la.mul(self._xform,la.translate(x,y))
        self._xform = la.mul(self._xform,la.scale(s,s))
        self._xform = la.mul(self._xform,la.translate(-x,-y))

        
    def addtag_closest(self, newtag, x, y, halo=None, start=None):
        """Add tag NEWTAG to item which is closest to pixel at X, Y.
        If several match take the top-most.
        All items closer than HALO are considered overlapping (all are
        closests). If START is specified the next below this tag is taken."""
        cx,cy = self._o2c(x,y)
        self.addtag(newtag, 'closest', cx, cy, halo, start)
    def addtag_enclosed(self, newtag, x1, y1, x2, y2):
        """Add tag NEWTAG to all items in the rectangle defined
        by X1,Y1,X2,Y2."""
        cx1,cy1 = self._o2c(x1,y1)
        cx2,cy2 = self._o2c(x2,y2)
        self.addtag(newtag, 'enclosed', cx1, cy1, cx2, cy2)
    def addtag_overlapping(self, newtag, x1, y1, x2, y2):
        """Add tag NEWTAG to all items which overlap the rectangle
        defined by X1,Y1,X2,Y2."""
        cx1,cy1 = self._o2c(x1,y1)
        cx2,cy2 = self._o2c(x2,y2)
        self.addtag(newtag, 'overlapping', cx1, cy1, cx2, cy2)
    def bbox(self, *args):
        """Return a tuple of X1,Y1,X2,Y2 coordinates for a rectangle
        which encloses all items with tags specified as arguments."""
        cbbox = self._getints(self.tk.call((self._w, 'bbox') + args)) or None
        if (cbbox):
            x1,y1 = self._c2o(cbbox[0],cbbox[1])
            x2,y2 = self._c2o(cbbox[2],cbbox[3])
            bbox = (x1,y1,x2,y2)
            return bbox
        return None
    def canvasxy(self, screenx, screeny):
        return self._w2o(screenx, screeny)
    def coords(self, *args):
        """Return a list of coordinates for the item given in ARGS."""
        args = tk._flatten(args)
        if len(args) > 1:
            cargs = tuple()
            cargs += (args[0],)
            for i in range(1,len(args),2):
                cx,cy = self._o2c(args[i],args[i+1])
                cargs += (cx,cy,)
        else:
            cargs = args
        # XXX Should use _flatten on args
        cc = [self.tk.getdouble(x) for x in
              self.tk.splitlist(self.tk.call((self._w, 'coords') + cargs))]
        for i in range(0,len(cc),2):
            x,y     = self._c2o(cc[i],cc[i+1])
            cc[i]   = x
            cc[i+1] = y
        return cc
    def _create(self, itemType, args, kw): # Args: (val, val, ..., cnf={})
        """Internal function."""
        args = tk._flatten(args)
        cnf = args[-1]
        if isinstance(cnf, (dict, tuple)):
            args = args[:-1]
        else:
            cnf = {}
        cargs= ()
        for i in range(0,len(args),2):
            cx,cy = self._o2c(args[i],args[i+1])
            cargs += (cx,cy,)
        return self.tk.getint(self.tk.call(
            self._w, 'create', itemType,
            *(cargs + self._options(cnf, kw))))
    def create_arc(self, *args, **kw):
        """Create arc shaped region with coordinates x1,y1,x2,y2."""
        return self._create('arc', args, kw)
    def create_bitmap(self, *args, **kw):
        """Create bitmap with coordinates x1,y1."""
        return self._create('bitmap', args, kw)
    def create_image(self, *args, **kw):
        """Create image item with coordinates x1,y1."""
        return self._create('image', args, kw)
    def create_line(self, *args, **kw):
        """Create line with coordinates x1,y1,...,xn,yn."""
        return self._create('line', args, kw)
    def create_oval(self, *args, **kw):
        """Create oval with coordinates x1,y1,x2,y2."""
        return self._create('oval', args, kw)
    def create_polygon(self, *args, **kw):
        """Create polygon with coordinates x1,y1,...,xn,yn."""
        return self._create('polygon', args, kw)
    def create_rectangle(self, *args, **kw):
        """Create rectangle with coordinates x1,y1,x2,y2."""
        return self._create('rectangle', args, kw)
    def create_text(self, *args, **kw):
        """Create text with coordinates x1,y1."""
        return self._create('text', args, kw)
    def create_window(self, *args, **kw):
        """Create window with coordinates x1,y1,x2,y2."""
        return self._create('window', args, kw)
    def find_closest(self, x, y, halo=None, start=None):
        """Return item which is closest to pixel at X, Y.
        If several match take the top-most.
        All items closer than HALO are considered overlapping (all are
        closests). If START is specified the next below this tag is taken."""
        cx,cy = self._o2c(x,y)
        return self.find('closest', cx, cy, halo, start)
    def find_enclosed(self, x1, y1, x2, y2):
        """Return all items in rectangle defined
        by X1,Y1,X2,Y2."""
        cx1,cy1 = self._o2c(x1,y1)
        cx2,cy2 = self._o2c(x2,y2)
        return self.find('enclosed', cx1, cy1, cx2, cy2)
    def find_overlapping(self, x1, y1, x2, y2):
        """Return all items which overlap the rectangle
        defined by X1,Y1,X2,Y2."""
        cx1,cy1 = self._o2c(x1,y1)
        cx2,cy2 = self._o2c(x2,y2)
        return self.find('overlapping', cx1, cy1, cx2, cy2)
    def itemcget(self, tagOrId, option):
        """Return the resource value for an OPTION for item TAGORID."""
        return self.tk.call(
            (self._w, 'itemcget') + (tagOrId, '-'+option))
    def itemconfigure(self, tagOrId, cnf=None, **kw):
        """Configure resources of an item TAGORID.

        The values for resources are specified as keyword
        arguments. To get an overview about
        the allowed keyword arguments call the method without arguments.
        """
        return self._configure(('itemconfigure', tagOrId), cnf, kw)
    itemconfig = itemconfigure
    # lower, tkraise/lift hide Misc.lower, Misc.tkraise/lift,
    # so the preferred name for them is tag_lower, tag_raise
    # (similar to tag_bind, and similar to the Text widget);
    # unfortunately can't delete the old ones yet (maybe in 1.6)
    def tag_lower(self, *args):
        """Lower an item TAGORID given in ARGS
        (optional below another item)."""
        self.tk.call((self._w, 'lower') + args)
    lower = tag_lower
    def move(self, *args):
        """Move an item TAGORID given in ARGS."""
        cargs = (args[0],)
        cx,cy = self._o2cv(args[1],args[2])
        cargs += (cx,cy,)
        self.tk.call((self._w, 'move') + cargs)
    def postscript(self, cnf={}, **kw):
        """Print the contents of the canvas to a postscript
        file. Valid options: colormap, colormode, file, fontmap,
        height, pageanchor, pageheight, pagewidth, pagex, pagey,
        rotate, witdh, x, y."""
        return self.tk.call((self._w, 'postscript') +
                    self._options(cnf, kw))
    def scale(self, *args):
        """Scale item TAGORID with XORIGIN, YORIGIN, XSCALE, YSCALE."""
        cx,cy = self._o2c(args[1],args[2])
        self.tk.call((self._w, 'scale') + (args[0],cx,cy,args[3],args[4]))
    def scan_mark(self, x, y):
        """Remember the current X, Y coordinates."""
        self.tk.call(self._w, 'scan', 'mark', x, y)
        self.scanmark = self._w2c(x,y)
    def scan_dragto(self, x, y, gain=10):
        """Adjust the view of the canvas to GAIN times the
        difference between X and Y and the coordinates given in
        scan_mark."""
        ox,oy = self._w2c(x,y)
        self.tk.call(self._w, 'scan', 'dragto', x, y, gain)
        dx,dy = ox - self.scanmark[0],oy - self.scanmark[1]
        super().move("fixed-to-window",-dx*gain,-dy*gain)
    def xview(self, *args):
        """Query and change the horizontal position of the view."""
        px,py = self._w2o(0,0)
        res = self.tk.call(self._w, 'xview', *args)
        if not args:
            return self._getdoubles(res)
        x,y = self._w2o(0,0)
        dx,dy = self._o2cv(x-px,y-py)
        super().move("fixed-to-window", dx, dy)
    def xview_moveto(self, fraction):
        """Adjusts the view in the window so that FRACTION of the
        total width of the canvas is off-screen to the left."""
        px,py = self._w2o(0,0)
        self.tk.call(self._w, 'xview', 'moveto', fraction)
        x,y = self._w2o(0,0)
        dx,dy = self._o2cv(x-px,y-py)
        super().move("fixed-to-window", dx, dy)
    def xview_scroll(self, number, what):
        """Shift the x-view according to NUMBER which is measured in "units"
        or "pages" (WHAT)."""
        px,py = self._w2o(0,0)
        self.tk.call(self._w, 'xview', 'scroll', number, what)
        x,y = self._w2o(0,0)
        dx,dy = self._o2cv(x-px,y-py)
        super().move("fixed-to-window", dx, dy)
    def yview(self, *args):
        """Query and change the vertical position of the view."""
        px,py = self._w2o(0,0)
        res = self.tk.call(self._w, 'yview', *args)
        if not args:
            return self._getdoubles(res)
        x,y = self._w2o(0,0)
        dx,dy = self._o2cv(x-px,y-py)
        super().move("fixed-to-window", dx, dy)
    def yview_moveto(self, fraction):
        """Adjusts the view in the window so that FRACTION of the
        total height of the canvas is off-screen to the top."""
        px,py = self._w2o(0,0)
        self.tk.call(self._w, 'yview', 'moveto', fraction)
        x,y = self._w2o(0,0)
        dx,dy = self._o2cv(x-px,y-py)
        super().move("fixed-to-window", dx, dy)
    def yview_scroll(self, number, what):
        """Shift the y-view according to NUMBER which is measured in
        "units" or "pages" (WHAT)."""
        px,py = self._w2o(0,0)
        self.tk.call(self._w, 'yview', 'scroll', number, what)
        x,y = self._w2o(0,0)
        dx,dy = self._o2cv(x-px,y-py)
        super().move("fixed-to-window", dx, dy)
