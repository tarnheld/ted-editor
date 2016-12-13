import tkinter as tk
from tkinter import font as tkFont
import dialogs as dg
import transformations as tf
import readfile_2 as rf

from bisect import bisect_left
from random import randint
        
class ElevationEditor(tk.Frame):
    '''An application for editing the height data of a TED file'''

    def __init__(self, master, trackapp=None, tokenlist=None, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        self.pack(side=tk.LEFT,expand=True,fill=tk.BOTH)
        self.trackapp = trackapp

        # app title
        version = "1.1.4a"
        self.master.title("Elevation Editor v."+ version)

        new_tags = self.bindtags() + ("all",)
        self.bindtags(new_tags)

        # token and brush data
        self._tokens = tuple()
        self._elevationProfiles  = []
        self._drag_data = {'x': 0, 'y': 0, 'items': [], 'feather': []}
        self._brush_data = {'x': 0, 'r': tk.DoubleVar(value=10), 'f0': tk.DoubleVar(value=10), 'f1': tk.DoubleVar(value=10)} #r = brush radius, f = feather radius
        self._brushes = tuple()
        self._feathers = tuple()
        self._leftfeather = False
        self._rightfeather = False
        self._feathermode2 = [tk.StringVar(), tk.StringVar()]
        self._mouselabel = {"x_coord": 0, "y_coord": 0, 'x':tk.StringVar(), 'z':tk.StringVar()}
        self._mousepos = {'x': 0, 'y': 0}
        self._widgetsize = (None, None)
        self._past = []
        self._future = []
        
        # set offset, scales
        self._high_x = 0
        self._high_y = 0
        self._low_x = 0
        self._low_y = 0
        self._first_z = 0
        self._brush_offset = self._high_x - self._low_x
        self._y_scale = -5.0
        self._x_scale = 1.0
        self._slopescale = -500
        self.isLoop = 0

        # set graphics
        self.line_width = 2      
        can_bg = "#6090CC"
        self.c_gridFill = "#4C74A4"    
        gCan_bg = "#303030"
        self.g_gridFill = "#202020"    
        self.tokenFill = "#FFFFFF"
        self.brushFill = "#5682B8"
        self.featherFill = "#5B89C2"



        ### MENUS ----------------------------------------------------------------------------------
        
        # create a toplevel menu and a frame
        self.menubar = tk.Menu(self)
        
        #create pulldown file menu
        filemenu = tk.Menu(self.menubar, tearoff = 0)
        filemenu.add_command(label="Load", command=self._load_ted)
        filemenu.add_command(label="Export", command=self._export_ted)
        filemenu.add_separator()
        #filemenu.add_command(label="Import elevation profile", command=self._import_ep)
        filemenu.add_command(label="Export elevation profile", command=self._export_ep)
        filemenu.add_separator()
        filemenu.add_command(label="Quit", command=self.quit)
        self.menubar.add_cascade(label="File", menu = filemenu)
        
        #create pulldown edit menu
        editmenu = tk.Menu(self.menubar, tearoff = 0)
        editmenu.add_command(label="Undo", command = self._undo)
        editmenu.add_command(label="Redo", command = lambda: self._undo(arg="redo"))
        self.menubar.add_cascade(label="Edit", menu = editmenu)

        #create pulldown help menu
        helpmenu = tk.Menu(self.menubar, tearoff = 0)
        helpmenu.add_command(label="Quick reference", command=self._quick_reference)
        self.menubar.add_cascade(label="Help", menu = helpmenu)

        #display the menu
        self.master.config(menu = self.menubar)

        #left panel
        self.leftPanel = dg.LeftPanel(self)


        
        ### CREATE CANVASES ---------------------------------------------------------------------------

        c_width = 600

        # create a token canvas
        self.canvas = tk.Canvas(self, width=c_width, bg = can_bg, height=400, bd=1, relief = "groove")
        new_tags = self.canvas.bindtags() + ("brusharea", "all")
        self.canvas.bindtags(new_tags)

        #create a graph canvas
        self.graphCanvas = tk.Canvas(self, width=c_width, bg = gCan_bg, height = 200, bd=1, relief = "groove")
        
        new_tags = self.graphCanvas.bindtags() + ("brusharea", "all")
        self.graphCanvas.bindtags(new_tags)

        # create the brush
        self._create_brush(self._brush_data['x'])



        ### SCROLLBARS ---------------------------------------------------------------------------------

        #add scrollbars
        self.hbar = tk.Scrollbar(self, orient="horizontal", command=self.xview)
        self.vbar = tk.Scrollbar(self, orient="vertical", command=self.yview)
        self.canvas.config(xscrollcommand = self.hbar.set, yscrollcommand = self.vbar.set)

        #add scalebar
        self.graphScaleBar = tk.Scrollbar(self, orient="vertical", command=self.graphScale)



        ### RULERS -----------------------------------------------------------------------------------

        self.r_width = 50
        
        #horizontal
        self.hruler = tk.Canvas(self, width=c_width, height=15, bd=1) #horizontal
        self.c_vruler = tk.Canvas(self, width=self.r_width, height=15, bd=1) #canvas
        self.g_vruler = tk.Canvas(self, width=self.r_width, height=15, bd=1) #graph



        ### BINDINGS ------------------------------------------------------------------------------------
        
        self.bind_class("brusharea", "<ButtonPress-1>", self.OnTokenButtonPress)
        self.bind_class("brusharea", "<ButtonRelease-1>", self.OnTokenButtonRelease)
        self.bind_class("brusharea", "<B1-Motion>", self.OnTokenMotion)
        self.bind_class("brusharea", "<Shift-B1-Motion>", self.ShiftOnTokenMotion)
        self.bind_class("brusharea", "<Motion>", self.OnMotion)
        self.bind_class("brusharea", '<MouseWheel>', self.MouseWheel)
        self.bind_class("brusharea", '<Shift-MouseWheel>', self.ShiftMouseWheel)
        self.bind_class("brusharea", '<Control-MouseWheel>', self.ControlMouseWheel)
        self.bind_class("brusharea", '<Shift-Control-MouseWheel>', self.ShiftControlMouseWheel)
        self.bind_class("brusharea", '<ButtonPress-3>', self.Button3Press)
        self.bind_class("brusharea", '<B3-Motion>', self.MouseScroll)
        self.bind('f', self.FButtonPress)
        self.bind('s', self.SButtonPress)
        self.bind('z', self.zkey)
        self.bind('<Alt-z>', self.altzkey)
        self.bind('<Control-z>', self._undo)
        self.bind('<Control-y>', self._undo)
        self.bind('<Left>', self._leftkey)
        self.bind('<Right>', self._rightkey)
        self.bind('<Configure>', self._my_configure)
        self.bind('<Return>', self._returnkey)  

     

        ### GEOMETRY MANAGEMENT ---------------------------------------------------------------------
        
        self.leftPanel.grid(row=0, column=0, rowspan=3, sticky="NS")
        self.canvas.grid(row=0, column=2, sticky = "WENS")
        self.graphCanvas.grid(row=2, column=2, sticky = "WE")
        self.hbar.grid(row=3, column=2, sticky = "WE")
        self.vbar.grid(row=0, column=3, sticky = "NS")
        self.graphScaleBar.grid(row=2, column=3, sticky = "NS")
        self.hruler.grid(row=1, column=2, sticky = "WE")
        self.c_vruler.grid(row=0, column=1, sticky = "NS")
        self.g_vruler.grid(row=2, column=1, sticky = "NS")

        self.columnconfigure(2, weight=1)
        self.rowconfigure(0, weight=1)
        


        ### OTHER ---------------------------------------------------------------------------------
 
        self.update_idletasks()
        self.focus_set()


    ### IMPORTING AND EXPORTING FILES -------------------------------------------------------------

    def _load_ted(self, structure=None):
        
        def delete_old(self):       
            #delete any old tokens, origins and brushes
            self._past = []
            self._future = []
            
            origins = self.canvas.find_withtag("origin")
            
            deletethis = self._tokens + origins + self._brushes + self._feathers
            for item in deletethis:
                self.canvas.delete(item) 

        def extract(self, structure):

            def _create_token(self, coord, color):
                '''Create a token at the given coordinate in the given color'''
                (x,y) = coord
                token = self.canvas.create_rectangle(x, y, x, y, 
                                        outline=color, width=1, fill=color, tags="token")
                        
                
            self.structure = s = structure
            self.isLoop = s.header['isloopcourse'][-1]
            tokenlist = s.tokens

            # get first_z           
            self._first_z = tokenlist[0][1]

            #normalize to 0
            normalized_tokens = [(x, z-self._first_z) for (x,z) in tokenlist]
            tokenlist = normalized_tokens

            #find highest and lowest x, y in tokens
            self._find_xy(tokenlist)

            # set brush offset
            if self.isLoop:
                self._brush_offset = self._high_x - self._low_x
            else:
                self._brush_offset = 0
            #self._y_scale = -5

            #draw brush
            self._create_brush(self._brush_data['x'])

            # draw origin
            self._create_guide(tokenlist)

            #draw tokens
            for (x, z) in tokenlist:
                _create_token(self, (x, z*self._y_scale), self.tokenFill)
            self._tokens = self.canvas.find_withtag('token')
            self._find_slope()
            self._scrollregion()

            # position brush
            self.brushChange(x = self.canvas.canvasx(0))
            self._brush_data['x'] = self.canvas.canvasx(0)

            #draw elevation profiles
            for ep in self._elevationProfiles:
                ep.draw()
            
        #load a .ted file
        if structure == None:
            structure = rf.load_TED()
        if structure != None:
            delete_old(self) 
            extract(self, structure)

    def _export_ted(self):
        tokenlist = self._gen_tokenlist()
        self.structure.mod = [token[1]+self._first_z for token in tokenlist]
        rf.export_TED(self.structure)

    def _import_ep(self):
        ep = rf.importElevationProfile()
        return ep
        #self._create_guide(ep)
        #self._order()

    def _export_ep(self):
        heightslist = self._gen_tokenlist()
        denormalized = [(x, y+self._first_z) for (x, y) in heightslist]
        rf.exportElevationProfile(denormalized)

    def _gen_tokenlist(self):
        tokenlist = []
        for token in self._tokens:
            y = self.canvas.coords(token)[1]/self._y_scale
            x = self.canvas.coords(token)[0]
            tokenlist.append((x, y))
        return tokenlist

    def _create_brush(self, x):
        '''Create a brush at x coordinate'''
        
        ranges = (range(0, 1), range(-1, 2))
        _range = ranges[self.isLoop]
        r = self._brush_data['r'].get()
        f0 = self._brush_data['f0'].get()
        f1 = self._brush_data['f1'].get()
        f_col = self.featherFill
        b_col = self.brushFill
        y0 = self.canvas.canvasy(-10000)
        y1 = self.canvas.canvasy(10000)
        for index in _range:
            x0 = (self._brush_offset*index) + x-r
            x1 = (self._brush_offset*index) + x+r
            feather = self.canvas.create_rectangle(x0-f0, y0, x1+f1, y1, width=0,
                fill = f_col, tags="feather")
            brush = self.canvas.create_rectangle(x0, y0, x1, y1, width=0,
                fill = b_col, tags="brush")

        self._brushes = self.canvas.find_withtag('brush')
        self._feathers = self.canvas.find_withtag('feather')






    ### UNDO, REDO, QUICK REFERENCE -------------------------------------------------------------------------------

    def _undo(self, event = None, arg="undo"):
        if event != None:
            if event.keysym == 'y':
                arg = "redo"
        if arg == "undo":
            try:            
                heightslist = self._past.pop()
                self._appendToHistory("future")
            except IndexError:
                heightslist = None
        else:
            try:           
                heightslist = self._future.pop()
                self._appendToHistory()
            except IndexError:
                heightslist = None
        if heightslist != None:
            c = self.canvas
            s = self._y_scale

            for i, token in enumerate(self._tokens):
                x, z = heightslist[i][0:2]
                c.coords(token, x, z*s, x, z*s)
                
            self._scrollregion()

    def _appendToHistory(self, arg="past"):
        if arg == "past":
            heightslist = self._past
        else:
            heightslist = self._future
            
        c = self.canvas
        s = self._y_scale
        h = []
        
        for token in self._tokens:
            x, z = c.coords(token)[0:2]
            h.append((x, z/s))
            
        heightslist.append(h)

    def _quick_reference(self):
        try:
            self._qr.destroy()
        except:
            None
        self._qr = dg.QuickReference(self)
        
        


    ### SLOPES GRIDS AND GUIDES -------------------------------------------------------------------------

    def _draw_slope_guides(self):

        def subdivide(p0, p1, subdivisions=3, step=None, _range=None):
            '''Subdivides the distance p1-p0'''
            
            def stepFinder(n, subdivisions):
                '''Finds the step size'''
                #Find if n is negative
                sign = 1 - (n < 0)*2

                #List containing the available step factors
                factors = (1, 2.5, 5)

                #Extract significand and potencia
                string = '%.1E' %(n/subdivisions)
                [significand, potencia] = [float(s) for s in string.split('E')]

                #Select step factor and return step size
                stepFactor = min(factors, key=lambda x:abs(x-abs(significand)))*sign
                step = stepFactor*10**potencia
                return step

            p0, p1 = sorted((p0, p1))
            n = p1-p0 
            if step==None:
                step = stepFinder(n, subdivisions)

            if _range==None:
                _range=(p0, p1)
            r0, r1 = _range[0], _range[1]
                
            offset = (r0//step)*step

            my_list = []
            i = 1
            while float(format(offset+i*step, 'g')) < r1:
                my_list.append(float(format(offset+i*step, 'g')))
                i += 1

            return my_list, step
        
        def frange(start, stop, step):
            i = start
            while i < stop:
                yield i
                i += step

        def horizontal_data(canvas, scale):
            h = canvas.winfo_height() / abs(scale)
            y0 = canvas.canvasy(0) / abs(scale)
            y1 = y0 + h

            return (h, y0, y1)

        def vertical_data(self, canvas):
            w = canvas.winfo_width()
            x0 = 0
            x1 = self._high_x
            if x1 == None or x1 < w:
                x1 = w

            return (w, x0, x1)
            
        light = "#C9C9C9"
        medium = "#909090"
        dark = "#202020"

        rulerfont = tkFont.Font(family="Helvetica", size=8)

        guideslist = (self.graphCanvas, self.canvas, self.hruler, self.c_vruler, self.g_vruler)
        for canvas in guideslist:
            canvas.delete("guide")

        (h, y0, y1) = horizontal_data(self.graphCanvas, self._slopescale)
        (w, x0, x1) = vertical_data(self, self.graphCanvas)

        #horizontal grahpCanvas lines
        (lines, res) = subdivide(y0, y1, subdivisions=5)
        for line in lines:
            y = line*self._slopescale
            if line == 0:
                _width=2
            else:
                _width=1
            self.graphCanvas.create_line(x0, y, x1, y, fill=self.g_gridFill, width=_width, tags="guide")
            #create text
            _text = '%s%%' %format(line*100, 'g')
            self.g_vruler.create_text(self.r_width, y, text=_text, anchor="e", tags="guide", font=rulerfont)
            
        #vertical lines
        i = 0
        while i <x1:
            if i%1000 == 0:
                _dash = None
            else:
                _dash = (4, 2)
            self.graphCanvas.create_line(i, -100, i, 100, fill=self.g_gridFill, dash = _dash, tags="guide")
            self.canvas.create_line(i, -10000, i, 10000, fill=self.c_gridFill, dash = _dash, tags="guide")
            #create text
            if i%500 == 0:
                y_text = y0*-self._slopescale+14
                self.hruler.create_text(i, 7.5, text="%d m" %i, tags="guide", font=rulerfont)
            i+=100
      
        #horizontal token canvas lines
        (h, y0, y1) = horizontal_data(self.canvas, self._y_scale)
        if self._low_y == None:
            r0, r1 = y0, y1
        else:
            r1 = -min((y0, -self._high_y))
            r0 = -max((y1, -self._low_y))
        
        #(lines, res) = subdivide(y0, y1, subdivisions=5, _range=(r0, r1))
        (lines, res) = subdivide(0, 200/self._y_scale, subdivisions=4, _range=(r0, r1))
        for line in lines:
            y = line*-self._y_scale
            if line == 0:
                _width=2
            else:
                _width=1
            self.canvas.create_line(x0, -y, x1, -y, fill=self.c_gridFill, width=_width, tags="guide")
            #create text
            _text = '%s m' %format(line, 'g')
            self.c_vruler.create_text(self.r_width, -y, text=_text, anchor="e", tags="guide", font=rulerfont)

        self._order()

        
    def _order(self):
        self.canvas.tag_lower("guide")
        self.canvas.tag_lower("ep")
        self.canvas.tag_lower("origin")
        self.canvas.tag_lower("brush")
        self.canvas.tag_lower("feather")
        

    def _find_slope(self, tokenlist=None):
        
        def _draw_slope(self, index=1, fill="red"):
            slopes = self._slopes
            slopescale = self._slopescale
            offset = 0
            for i in range(0, len(slopes)-1):
                x = [slopes[i+n][0] for n in (0, 1)]
                y = [slopes[i+n][index]*slopescale for n in (0, 1)]
                self.graphCanvas.create_line(x[0], y[0], x[1], y[1],
                    fill=fill, width=self.line_width, tags="slope")        

        def _slope(self, x0, y0, x1, y1):
            delta_x = x1 - x0
            delta_y = y1 - y0
            slope = delta_y / delta_x
            return slope

        self.graphCanvas.delete("slope")
        
        self._slopes = []
        if tokenlist == None:
            tokenlist = self._gen_tokenlist()

        tokenlen = len(tokenlist)

        for index, token in enumerate(tokenlist):
            x = token[0]
            y = token[1]

            if index == 0:
                slope_a = 0
                prev_slope = 0

            else:
                prev_x = tokenlist[index-1][0]
                prev_y = tokenlist[index-1][1]
                slope_a = _slope(self, prev_x, prev_y, x, y)
                               
            if index != tokenlen-1:   
                next_x = tokenlist[index+1][0]
                next_y = tokenlist[index+1][1]
                slope_b = _slope(self, x, y, next_x, next_y)

            else:
                slope_b = 0
            
            slope = (slope_a + slope_b) / 2
            delta_slope = slope - prev_slope
            prev_slope = slope
            self._slopes.append((x, slope, delta_slope))

        if len(self._slopes) > 0:
            _draw_slope(self, 1, "#FF0000")
            _draw_slope(self, 2, "#0090FF")


    def _create_guide(self, tokenlist, tag="origin"):
        prev_xy = None
        for index, token in enumerate(tokenlist):
            (x, y) = token[0:2]
            y *= self._y_scale
            if prev_xy != None:
                self.canvas.create_line(prev_xy[0], prev_xy[1], x, y,
                                        fill = self.c_gridFill, width = self.line_width, tags=tag)
            prev_xy = (x, y)



    ### MOUSE COMMANDS --------------------------------------------------------------------------------------------
        
    def OnMotion(self, event):
        '''Handle moving the brush'''
        self.brushChange(x = self.canvas.canvasx(event.x)) 
        self._brush_data['x'] = self.canvas.canvasx(event.x)
        self.MouseLabelChange(event)
        if self.trackapp:
            l = self._brush_data['x']/self._x_scale
            r = self._brush_data['r'].get()/self._x_scale
            f0 = self._brush_data['f0'].get()/self._x_scale
            f1 = self._brush_data['f1'].get()/self._x_scale
            #self.trackapp.drawTrackIndicator(self.canvas.canvasx(event.x)/self._x_scale)
            self.trackapp.drawTrackOffsetIndicator(l-r,l+r,f0,f1)

    def OnTokenButtonPress(self, event):
        '''Begin drag of an object'''
        
        # record the item and its location
        self._appendToHistory()
        self._future = []
        #get brush data
        ranges = (range(0, 1), range(-1, 2))
        _range = ranges[self.isLoop]
        x = self.canvas.canvasx(event.x) 
        f0 = self._brush_data['f0'].get()
        f1 = self._brush_data['f1'].get()
        o = self._brush_offset
        #get drag data
        for i in _range:
            self.get_drag_data(x, i, f=(f0, f1))
            
        self._drag_data["x"] = self.canvas.canvasx(event.x)
        self._drag_data["y"] = event.y

    def OnTokenButtonRelease(self, event):
        '''End drag of an object'''
        
        # reset the drag information
        self._drag_data["items"] = []
        self._drag_data["feather"] = []
        self._drag_data["x"] = 0
        self._drag_data["y"] = 0
        self.OnMotion(event)
        self._scrollregion()

    def OnTokenMotion(self, event, factor=1):
        '''Handle dragging of an object'''
        
        #initial definitions
        ranges = (range(0, 1), range(-1, 2))
        _range = ranges[self.isLoop]
        o = self._brush_offset
        r = self._brush_data['r'].get()
        f0 = self._brush_data['f0'].get()
        f1 = self._brush_data['f1'].get()
        f0_mode = self._feathermode2[0].get()
        f1_mode = self._feathermode2[1].get()
        
        # compute how much this object has moved
        delta_y = (event.y - self._drag_data["y"])*factor
        
        # move brush and feather objects
        for index, i in enumerate(_range):
            x = (o*i) + self._drag_data['x']

            #move brush objects
            for item in self._drag_data['items'][index]:
                self.canvas.move(item, 0, delta_y)
                    
            # move feathered objects
            for item in self._drag_data['feather'][index]:
                token_x = self.canvas.coords(item)[0]
                delta_x = abs(x-token_x)-r
                if token_x < x:
                    feather = tf.feather(f0, f0_mode, delta_x)
                else:
                    feather = tf.feather(f1, f1_mode, delta_x)
                self.canvas.move(item, 0, delta_y*feather)               

        # record the new position
        self._drag_data["y"] = event.y
        self.MouseLabelChange(event, mode="y", factor=factor)

    def ShiftOnTokenMotion(self, event):
        '''Handle shift-dragging of an object'''
        self.OnTokenMotion(event, factor=0.1)

    def MouseWheel(self, event, modes = ('r',), step=8):
        '''change brush / feather radius'''
        for mode in modes:
            r = self._brush_data[mode].get()
            if event.delta == -120:
                r -= step
                if r < 0:
                    r = 0
            elif event.delta == 120:
                r += step
                if r > 1000:
                    r = 1000
            if mode == 'r':
                self.brushChange(r = r)
            elif mode =='f0':
                self.brushChange(f0 = r)
            elif mode == 'f1':
                self.brushChange(f1 = r)
            self._brush_data[mode].set(r)
        self.MouseLabelChange(event)

    def ShiftMouseWheel(self, event):
        self.MouseWheel(event, step=1)

    def ControlMouseWheel(self, event, step=8):
        '''change feather radius'''
        if self._leftfeather-self._rightfeather == 0:
            self.MouseWheel(event, modes=('f0', 'f1'), step=step)
        elif self._leftfeather == 1:
            self.MouseWheel(event, modes=('f0',), step=step)
        else:
            self.MouseWheel(event, modes=('f1',), step=step)

    def ShiftControlMouseWheel(self, event):
        self.ControlMouseWheel(event, step=1)

    def Button3Press(self, event):
        self.canvas.scan_mark(event.x, event.y)
        self.c_vruler.scan_mark(0, event.y)
        for canvas in (self.graphCanvas, self.hruler):
            canvas.scan_mark(event.x, 0)

    def MouseScroll(self, event):
        g = 2
        self.canvas.scan_dragto(event.x, event.y, gain=g)
        self.c_vruler.scan_dragto(0, event.y, gain=g)
        for canvas in (self.graphCanvas, self.hruler):
            canvas.scan_dragto(event.x, 0, gain=g)

    def get_drag_data(self, x, index=0, f=None):
        '''Select objects'''
        
        # record the item and its location
        r = self._brush_data['r'].get()
        o = self._brush_offset
        y0 = self.canvas.canvasy(-10000)
        y1 = self.canvas.canvasy(10000)  
        start = (o*index)+x-r
        end = (o*index)+x+r
        brush = [i for i in self.canvas.find_overlapping(start, y0, end, y1) if i in self._tokens]
        self._drag_data['items'].append(brush)
        if f != None:
            feather = [i for i in self.canvas.find_overlapping(start-f[0], y0, end+f[1], y1) if i in self._tokens and i not in brush]    
            self._drag_data["feather"].append(feather)

    def brushChange(self, x=None, r=None, f0=None, f1=None):
        '''Handles motion and resizing of brush and feather'''
        
        if r == None:
            r = self._brush_data['r'].get()
        if f0 == None:
            f0 = self._brush_data['f0'].get()
        if f1 == None:
            f1 = self._brush_data['f1'].get()
        if x == None:
            x = self._brush_data['x']

        ranges = (range(0, 1), range(-1, 2))
        _range = ranges[self.isLoop]
            
        y0 = self.canvas.canvasy(-10000)
        y1 = self.canvas.canvasy(10000)
        o = self._brush_offset
        
        for brush, index in enumerate(_range):
            start = (o*index)+x-r
            end = start+(2*r)
            self.canvas.coords(self._brushes[brush], start, y0, end, y1)
            self.canvas.coords(self._feathers[brush], start-f0, y0, end+f1, y1)

    def MouseLabelChange(self, event, mode="all", factor=1):
        '''Handles the display of the brush data'''
        
        if mode == "all": #OnMotion
            x_coord = self.canvas.canvasx(event.x)
            y_coord = y = self.canvas.canvasy(event.y)
            
        else: #OnTokenMotion
            x_coord = self._mouselabel['x_coord']
            y_coord = self._mouselabel['y_coord']
            y = self.canvas.canvasy(event.y) - y_coord
            
        r = self._brush_data['r'].get()
        f0 = self._brush_data['f0'].get()
        f1 = self._brush_data['f1'].get()

        x = '%.3f m' % x_coord
        z = '%.3f m' %(y*factor/self._y_scale)

        self._mouselabel['x'].set(x)
        self._mouselabel['z'].set(z)
        
        #save mouselabel
        self._mouselabel['x_coord'] = x_coord
        self._mouselabel['y_coord'] = y_coord



    ### TRANSFORMATIONS COMMANDS ----------------------------------------------------------------

    def _contrastDialog(self):
        dialog = dg.ContrastDialog(self)
        self.wait_window(dialog.top)

    def _contrast(self, weight=0):
        '''Scale all height points'''
        self._appendToHistory()
        self._future = []

        for item in self._tokens:
            x, z = self.canvas.coords(item)[0:2]
            z *= weight
            self.canvas.coords(item, x, z, x, z)
            
        #draw slope
        self._scrollregion()

    def _roughenDialog(self):
        dialog = dg.RoughenDialog(self)
        self.wait_window(dialog.top)
        
    def _roughen(self, potencia = 4, magnitude = 0.02, variation = 100):
        '''Apply small random elevation changes to all height points'''
        self._appendToHistory()
        self._future = []
        scale = magnitude*self._y_scale
        v = variation

        for item in self._tokens:
            delta_y = ((randint(0, v)/v)**potencia)*scale
            neg = randint(0, 1)
            if neg:
                delta_y*=-neg
            self.canvas.move(item, 0, delta_y)
            
        #draw slope
        self._scrollregion()

    def _smoothenDialog(self):
        dialog = dg.SmoothenDialog(self)
        self.wait_window(dialog.top)
        
    def _smoothen(self, tol=0.04, minL=6, maxL=100):
        '''Apply a smoothening algorithm to all height points'''
        self._appendToHistory()
        self._future = []
        c = self.canvas
        heightsList = [c.coords(i)[0:2] for i in self._tokens]

        rounded = tf.smoothen(heightsList, tol, minL, maxL)
        
        for item in self._tokens:
            x, z = rounded.pop(0)[0:2]
            c.coords(item, x, z, x, z)
            
        #draw slope
        self._scrollregion()

    def _translateDialog(self):
        dialog = dg.TranslateDialog(self)
        self.wait_window(dialog.top)

    def _translate(self, z=0):
        '''Translate all heightpoints along z axis'''
        self._appendToHistory()
        self._future = []
        
        for item in self._tokens:
            delta_z = z*self._y_scale
            self.canvas.move(item, 0, delta_z)
            
        #draw slope
        self._scrollregion()

    def _resampleDialog(self):
        dialog = dg.ResampleDialog(self)
        self.wait_window(dialog.top)

    def _resample(self, minstep=2):
        '''Change height data resolution'''
        struct = self.structure
        heightsList = self._gen_tokenlist()
        
        resampled = []
        
        for bank in struct.banks:
            unk = bank['unk']
            bankheights = heightsList[unk:unk+bank['divNum']+1]
            bank['unk'] = len(resampled)
            resampled+=(tf.resample(bankheights, minstep))
            bank['divNum'] = len(resampled)-bank['unk']

        resampled.append(heightsList[-1])
        struct.tokens = [(token[0], token[1]+self._first_z) for token in resampled]
        struct.mod = [token[1] for token in struct.tokens]
        struct.update_data()

        self._load_ted(struct)



    ### KEY COMMANDS ------------------------------------------------------------------------------

    def FButtonPress(self, event):
        '''Flatten section covered by brush'''
        self._appendToHistory()
        self._future = []
        x = self._brush_data['x']
        self.get_drag_data(x)
        
        '''Transformation'''
        try:
            A = self._drag_data['items'][0].pop(0)
            A = self.canvas.coords(A)
            B = self._drag_data['items'][0].pop(-1)
            B = self.canvas.coords(B)

            for P in self._drag_data['items'][0]:
                new_xy = tf.flatten(A, B, self.canvas.coords(P))
                self.canvas.coords(P, new_xy)
        except IndexError:
            print('IndexError')

        self._drag_data['items'] = []
        # draw slope
        self._find_slope()

    def SButtonPress(self, event):
        def _append_slope(self, item):
            coords = self.canvas.coords(item)
            slope = [i[1] for i in self._slopes if i[0] == coords[0]][0]
            item = (coords[0], coords[1], slope*self._y_scale)
            return item
        
        '''Select objects'''
        self._appendToHistory()
        self._future = []
        x = self._brush_data['x']
        self.get_drag_data(x)

        '''Transformation'''
        try:
            A = self._drag_data['items'][0].pop(0)
            A = _append_slope(self, A)
            B = self._drag_data['items'][0].pop(-1)
            B = _append_slope(self, B)

            for P in self._drag_data['items'][0]:
                new_xy = tf.spline(A, B, self.canvas.coords(P))
                self.canvas.coords(P, new_xy)
        except IndexError:
            None

        self._drag_data['items'] = []
        # draw slope
        self._find_slope()

    def _returnkey(self, event):
        self.focus_set()

    def zkey(self, event):
        self.Zoom(zoom=1.2)

    def altzkey(self, event):
        self.Zoom(zoom=1/1.2)

    def Zoom(self, zoom=2, axis='z'):
        '''Zoom in and out on the X or Z axis'''
        
        if axis == 'x':
            index = 0
            self._x_scale *= zoom
        else:
            index = 1
            self._y_scale *= zoom

        for item in self.canvas.find_withtag('token'):
            coords = self.canvas.coords(item) #get the coords of the item
            coords[index] *= zoom #scale the coordinate specified by the keystroke
            x, z = coords[0], coords[1]
            self.canvas.coords(item, x, z, x, z) #apply the new coordinates

        for item in self.canvas.find_withtag('origin')+self.canvas.find_withtag('ep'):
            coords = self.canvas.coords(item)
            coords[1] *= zoom
            coords[3] *= zoom
            self.canvas.coords(item, coords[0], coords[1], coords[2], coords[3])

        self._scrollregion()

    def _leftkey(self, event):
        self._rightfeather = False

        if self._leftfeather:
            self._leftfeather = False
        else:
            self._leftfeather = True

    def _rightkey(self, event):
        self._leftfeather = False

        if self._rightfeather:
            self._rightfeather = False
        else:
            self._rightfeather = True



    ### SCROLLING AND RESIZING -------------------------------------------------------------------

    def _my_configure(self, event):
        self.pack(side=tk.LEFT,expand=True,fill=tk.BOTH)
        self.after(10, self._resize)

    def _resize(self):
        '''Handle resized window'''
        self.update_idletasks()
        h = self.winfo_height()
        w = self.winfo_width()
        if (h, w) != self._widgetsize:
            self._widgetsize = (h, w)
            self._scrollregion()

    def graphScale(self, *args):
        None

    def xview(self, *args):
        self.canvas.xview(*args)
        self.hruler.xview(*args)
        self.graphCanvas.xview(*args)

    def yview(self, *args):
        self.canvas.yview(*args)
        self.c_vruler.yview(*args)

    def _find_xy(self, tokenlist=None):
        '''Find highest and lowest x and y in tokens'''
        self._high_x = None
        self._high_y = None
        self._low_x = None
        self._low_y = None
        if tokenlist == None:
            tokenlist = self._gen_tokenlist()
        for token in tokenlist:
            y = token[1]
            x = token[0]
            if self._high_x == None or x > self._high_x:
                self._high_x = x
            if self._high_y == None or y > self._high_y:
                self._high_y = y
            if self._low_x == None or x < self._low_x:
                self._low_x = x
            if self._low_y == None or y < self._low_y:
                self._low_y = y

    def _scrollregion(self, tokenlist = None):
        self._find_xy(tokenlist = tokenlist)
        bbox = self.canvas.bbox("token")
        gh = self.graphCanvas.winfo_height()
        ch = self.canvas.winfo_height()
        cw = self.canvas.winfo_width()

        if bbox != None:
            bbox = list(bbox)
            bbox[0]-=50
            bbox[1]-=200
            bbox[2]+=50
            bbox[3]+=200            
        else:
            bbox = (-50, -0.5*ch+3, cw-56, 0.5*ch-3)
            
        self.canvas.config(scrollregion=(bbox))
        self.c_vruler.config(scrollregion=(0, bbox[1], 0, bbox[3]))
        self.graphCanvas.config(scrollregion=(bbox[0], -0.5*gh+3, bbox[2], 0.5*gh-3))
        self.g_vruler.config(scrollregion=(0, -0.5*gh+3, 0, 0.5*gh-3))    
        self.hruler.config(scrollregion=(bbox[0], 0, bbox[2], 0))
            
        self._draw_slope_guides()
        self._find_slope()

    

if __name__ == "__main__":
    app = ElevationEditor()  
    app.mainloop()
