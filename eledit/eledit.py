import tkinter as tk
import transformations as tf
import readfile as rf
from bisect import bisect_left

from random import randint

class ContrastDialog:

    def __init__(self, parent):

        top = self.top = tk.Toplevel(parent)
        self.ElevationEditor = parent

        tk.Label(top, text='Contrast factor (0 to flatten)').grid(sticky="w")

        self.e = tk.Entry(top)
        self.e.grid(sticky="we", padx=5)
        self.e.focus()

        b = tk.Button(top, text='OK', command=self.ok)
        b.grid(sticky="we", pady=5)

    def ok(self):
        try:
            z = float(self.e.get())
        except:
            z = 0
        self.ElevationEditor._contrast(z)   
        self.top.destroy()

class RoughenDialog:

    def __init__(self, parent):

        top = self.top = tk.Toplevel(parent)
        self.ElevationEditor = parent

        self.entries = [['Potencia (float)', tk.StringVar(value='4.0')],
                   ['Magnitude (float)', tk.StringVar(value='0.02')],
                   ['Variation (int)', tk.StringVar(value='100')]]

        for entry in self.entries:
            tk.Label(top, text=entry[0]).grid(sticky='w')
            e = tk.Entry(top, textvariable = entry[1])
            e.grid(sticky="we", padx=5)

        b = tk.Button(top, text='OK', command=self.ok)
        b.grid(sticky="we", pady=5)
        top.focus_set()

    def ok(self):
        e = self.entries
        try:
            pot = float(e[0][1].get())
            mag = float(e[1][1].get())
            var = int(e[2][1].get())
            self.ElevationEditor._roughen(pot, mag, var)
        except:
            print('Exception raised.')
        self.top.destroy()

class SmoothenDialog:

    def __init__(self, parent):

        top = self.top = tk.Toplevel(parent)
        self.ElevationEditor = parent

        self.entries = [['Delta slope tolerance (float)', tk.StringVar(value='0.05')],
                   ['Min segment length (int)', tk.StringVar(value='6')],
                   ['Max segment length (int)', tk.StringVar(value='60')]]

        for entry in self.entries:
            tk.Label(top, text=entry[0]).grid(sticky='w')
            e = tk.Entry(top, textvariable = entry[1])
            e.grid(sticky="we", padx=5)

        b = tk.Button(top, text='OK', command=self.ok)
        b.grid(sticky="we", pady=5)
        top.focus_set()

    def ok(self):
        e = self.entries
        try:
            tol = float(e[0][1].get())
            minL = int(e[1][1].get())
            maxL = int(e[2][1].get())
            self.ElevationEditor._smoothen(tol, minL, maxL)
        except:
            print('Exception raised.')
        self.top.destroy()

class TranslateDialog:

    def __init__(self, parent):

        top = self.top = tk.Toplevel(parent)
        self.ElevationEditor = parent

        tk.Label(top, text='Raise by (m)').grid(sticky="w")

        self.e = tk.Entry(top)
        self.e.grid(sticky="we", padx=5)
        self.e.focus()

        b = tk.Button(top, text='OK', command=self.ok)
        b.grid(sticky="we", pady=5)

    def ok(self):
        try:
            z = float(self.e.get())
        except:
            z = 0
        self.ElevationEditor._translate(z)   
        self.top.destroy()

class ElevationEditor(tk.Frame):
    '''Illustrate how to drag items on a Tkinter canvas'''

    def __init__(self, master, trackapp=None, tokenlist=None, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        self.pack(side=tk.LEFT,expand=True,fill=tk.BOTH)
        self.trackapp = trackapp
        

        # app title
        version = "1.0.9"
        self.master.title("Elevation Editor v."+ version)

        new_tags = self.bindtags() + ("all",)
        self.bindtags(new_tags)

        # this data is used to keep track of an 
        # item being dragged
        self._drag_data = {"x": 0, "y": 0, "items": [], "feather": []}
        self._brush_data = {"x": 0, "brush": 10, "feather": 10, "f_mode": 0}
        self._feathermode = tk.IntVar()
        self._mouselabel = {"x_coord": 0, "y_coord": 0}
        self._mousepos = {'x': 0, 'y': 0}
        self._widgetsize = (None, None)
        self._past = []
        self._future = []
        
        # set offset, scales
        self._high_x = 0
        self._high_y = 0
        self._low_x = 0
        self._low_y = 0
        self._first_y = 0
        self._brush_offset = self._high_x - self._low_x
        self._y_scale = -5.0
        self._x_scale = 1.0
        self._slopescale = -500

        # set graphics
        self.line_width = 2
        
        can_bg = "#6090CC"
        self.c_gridFill = "#4C74A4"
        
        gCan_bg = "#303030"
        self.g_gridFill = "#202020"
        
        self.tokenFill = "#FFFFFF"
        self.brushFill = "#5682B8"
        self.featherFill = "#5B89C2"


        ### MENUS ###
        
        # create a toplevel menu and a frame
        self.menubar = tk.Menu(self)
        
        #create pulldown file menu
        filemenu = tk.Menu(self.menubar, tearoff = 0)
        filemenu.add_command(label="Load", command=self._load_ted)
        filemenu.add_command(label="Export", command=self._export_ted)
        filemenu.add_separator()
        filemenu.add_command(label="Quit", command=self.quit)
        self.menubar.add_cascade(label="File", menu = filemenu)
        
        #create pulldown edit menu
        editmenu = tk.Menu(self.menubar, tearoff = 0)
        editmenu.add_command(label="Undo", command = self._undo)
        editmenu.add_command(label="Redo", command = lambda: self._undo(arg="redo"))
        editmenu.add_separator()

        #feather submenu
        feathermenu = tk.Menu(editmenu, tearoff = 0)
        feathermenu.add_radiobutton(label="Wave", var=self._feathermode, value = 0)
        feathermenu.add_radiobutton(label="Linear", var=self._feathermode, value = 1)
        feathermenu.add_radiobutton(label="Bowl", var=self._feathermode, value = 2)
        feathermenu.add_radiobutton(label="Plateu", var=self._feathermode, value = 3)
        editmenu.add_cascade(label="Feather", menu = feathermenu)
     
        #transform submenu
        transformmenu = tk.Menu(editmenu, tearoff = 0)
        transformmenu.add_command(label="Contrast", command =self._contrastDialog)
        transformmenu.add_command(label="Roughen", command=self._roughenDialog)
        transformmenu.add_command(label="Smoothen", command=self._smoothenDialog)      
        transformmenu.add_command(label="Translate", command =self._translateDialog)
        editmenu.add_cascade(label="Transform", menu = transformmenu)
        
        self.menubar.add_cascade(label="Edit", menu = editmenu)

        #display the menu
        self.master.config(menu = self.menubar)


        ### CREATE LEFT PANEL ###

        # create left panel canvas
        self.leftPanel = tk.Frame(self)

        # create brush data canvas
        l = tk.LabelFrame(self.leftPanel, text = "Brush Data")
        c = tk.Canvas(l, width=200, height=80)
        _text = ('--- m\n'*4)[:-1]
        _labels = 'x coordinate:\nz coordinate:\nbrush radius:\nfeather radius:'
        c.create_text(15, 10, anchor="nw", text = _labels)
        c.create_text(180, 10, anchor="ne", text=_text, justify="right", tags = "mouselabel")
        self._brush_data['canvas'] = c
        l.grid(row=0, padx=5, pady=5, sticky="we")
        c.grid(row=0)

        #add radiobuttons for feather options
        rframe = tk.Frame(l)
        rframe.grid(sticky='w')
        for i, item in enumerate(['Wave', 'Linear', 'Bowl', 'Plateau']):
            rb = tk.Radiobutton(rframe, text=item, variable=self._feathermode, value=i)
            rb.grid(row=i//2, column=i%2, padx=2, pady=2, sticky='w')

        # create transformations buttons
        l = tk.LabelFrame(self.leftPanel, text = "Transform")
        buttons = [('Contrast', self._contrastDialog),
                   ('Roughen', self._roughenDialog),
                   ('Smoothen', self._smoothenDialog),
                   ('Translate', self._translateDialog)]

        for button in buttons:
            b = tk.Button(l, text=button[0], command = button[1])
            b.grid(padx=5, pady=2, sticky="we")
        l.grid(padx=5, pady=5, ipady=2, sticky="we")

        # create instructions panel
        l = tk.LabelFrame(self.leftPanel, text = "Commands")

        messages = ('Use brush: click + drag',
                    'Brush radius: mousewheel',
                    'Feather radius: ctrl + mousewheel',
                    'Panning: right-click + drag',
                    'Small incr. modifier: shift + [cmd]',
                    'Flatten section: F',
                    'Smoothen section: S',
                    'Zoom in: Z',
                    'Zoom out: alt + Z',
                    'Undo: ctrl + Z',
                    'Redo: ctrl + X')

        for message in messages:
            tk.Label(l, text = message, wraplength=200, anchor='w', justify='left').grid(sticky='we', padx=2)

        l.grid(padx=5, pady=5, ipady=2, sticky="we")
        
        ### CREATE CANVASES ###

        c_width = 600

        # create a token canvas
        self.canvas = tk.Canvas(self, width=c_width, bg = can_bg, height=400, bd=1, relief = "groove")
        new_tags = self.canvas.bindtags() + ("brusharea", "all")
        self.canvas.bindtags(new_tags)

        #create a graph canvas
        self.graphCanvas = tk.Canvas(self, width=c_width, bg = gCan_bg, height = 200, bd=1, relief = "groove")
        
        new_tags = self.graphCanvas.bindtags() + ("brusharea", "all")
        self.graphCanvas.bindtags(new_tags)

        ### SCROLLBARS ###

        #add scrollbars
        self.hbar = tk.Scrollbar(self, orient="horizontal", command=self.xview)
        self.vbar = tk.Scrollbar(self, orient="vertical", command=self.yview)
        self.canvas.config(xscrollcommand = self.hbar.set, yscrollcommand = self.vbar.set)

        #add scalebar
        self.graphScaleBar = tk.Scrollbar(self, orient="vertical", command=self.graphScale)


        ### RULERS ###

        self.r_width = 50
        
        #horizontal
        self.hruler = tk.Canvas(self, width=c_width, height=15, bd=1) #horizontal
        self.c_vruler = tk.Canvas(self, width=self.r_width, height=15, bd=1) #canvas
        self.g_vruler = tk.Canvas(self, width=self.r_width, height=15, bd=1) #graph

        # add bindings
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
        self.bind('<Configure>', self._my_configure)

        # create brushes and mouselabel
        self._create_brush(self._brush_data['x'])   
     

        ### GEOMETRY MANAGEMENT ###
        
        self.leftPanel.grid(row=0, column=0, rowspan=3, sticky="NS")
        self.canvas.grid(row=0, column=2, sticky = "WENS")
        self.graphCanvas.grid(row=2, column=2, sticky = "WE")
        self.hbar.grid(row=3, column=2, sticky = "WE")
        self.vbar.grid(row=0, column=3, sticky = "NS")
        self.graphScaleBar.grid(row=2, column=3, sticky = "NS")
        self.hruler.grid(row=1, column=2, sticky = "WE")
        self.c_vruler.grid(row=0, column=1, sticky = "NS")
        self.g_vruler.grid(row=2, column=1, sticky = "NS")
        

        ### OTHER ###

        self.update_idletasks()
        self.focus_set()
        self.columnconfigure(2, weight=1)
        self.rowconfigure(0, weight=1)
    def delete_old(self):       
        #delete any old tokens and origins
        self._past = []
        self._future = []
        tokens = self.canvas.find_withtag("token")
        for token in tokens:
            self.canvas.delete(token)
            
        origins = self.canvas.find_withtag("origin")
        for origin in origins:
            self.canvas.delete(origin)

    def extract(self, structure):
        def _create_origin(self, tokenlist):
            prev_xy = None            
            for index, token in enumerate(tokenlist):
                (x, y) = token[0:2]
                y *= self._y_scale
                y += self._first_y
                if prev_xy != None:
                    self.canvas.create_line(prev_xy[0], prev_xy[1], x, y,
                                            fill = self.c_gridFill, width = self.line_width, tags="origin")
                prev_xy = (x, y)
        def _create_token(self, coord, color):
            '''Create a token at the given coordinate in the given color'''
            (x,y) = coord
            y += self._first_y
            token = self.canvas.create_rectangle(x, y, x, y, 
                                                 outline=color, width=1, fill=color, tags="token")
            self._tokens.append(token)
                        
                
        self.structure = s = structure
        tokenlist = s['tokens']

        #find highest and lowest x, y in tokens
        self._find_xy(tokenlist)
        
        # set brush offset
        self._brush_offset = self._high_x - self._low_x
        self._y_scale = -5
        
        # get first y           
        first_y = tokenlist[0][1]
        self._first_y = 0 - first_y * self._y_scale
        
        # draw origin
        _create_origin(self, tokenlist)
        
        #draw tokens
        self._tokens = []
        for token in tokenlist:
            _create_token(self, (token[0], token[1]*self._y_scale), self.tokenFill)
        self._find_slope()
        self._scrollregion()

        # position brush
        self.brushChange(x = self.canvas.canvasx(0))
        self._brush_data['x'] = self.canvas.canvasx(0)

    def _load_ted(self):
        #load a .ted file
        structure = rf.load_struct()
        if structure != None:
            self.delete_old()
            self.extract(structure)
    def get_heightlist(self):
        tokenlist = self._gen_tokenlist()
        return [token[1]-(self._first_y/self._y_scale) for token in tokenlist]
    def update_structure(self):
        self.structure['mod'] = self.get_heightlist()
        return self.structure
    def _export_ted(self):
        self.update_structure()
        rf.export_struct(self.structure)

    def _undo(self, event = None, arg="undo"):
        if event != None:
            if not event.keysym == 'z':
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
            tokens = c.find_withtag("token")
            for i, token in enumerate(tokens):
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
        tokens = c.find_withtag("token")
        h = []
        for token in tokens:
            x, z = c.coords(token)[0:2]
            h.append((x, z/s))
        heightslist.append(h)
        

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

        guideslist = (self.graphCanvas, self.canvas, self.hruler, self.c_vruler, self.g_vruler)
        for canvas in guideslist:
            canvas.delete("guide")

        (h, y0, y1) = horizontal_data(self.graphCanvas, self._slopescale)
        (w, x0, x1) = vertical_data(self, self.graphCanvas)

        #horizontal grahpCanvas lines
        (lines, res) = subdivide(y0, y1)
        for line in lines:
            y = line*self._slopescale
            if line == 0:
                _width=2
            else:
                _width=1
            self.graphCanvas.create_line(x0, y, x1, y, fill=self.g_gridFill, width=_width, tags="guide")
            #create text
            _text = '%s%%' %format(line*100, 'g')
            self.g_vruler.create_text(self.r_width, y, text=_text, anchor="e", tags="guide")
            
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
                self.hruler.create_text(i, 7.5, text="%d m" %i, tags="guide")
            i+=100
      
        #horizontal token canvas lines
        (h, y0, y1) = horizontal_data(self.canvas, self._y_scale)
        if self._low_y == None:
            r0, r1 = y0, y1
        else:
            r1 = -min((y0, -self._high_y))
            r0 = -max((y1, -self._low_y))
        
        (lines, res) = subdivide(y0, y1, subdivisions=5, _range=(r0, r1))
        for line in lines:
            y = line*-self._y_scale
            if line == 0:
                _width=2
            else:
                _width=1
            self.canvas.create_line(x0, -y, x1, -y, fill=self.c_gridFill, width=_width, tags="guide")
            #create text
            _text = '%s m' %format(line, 'g')
            self.c_vruler.create_text(self.r_width, -y, text=_text, anchor="e", tags="guide")

        self.canvas.tag_lower("guide")
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
                               
            if index != len(tokenlist)-1:   
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
            

    def _gen_tokenlist(self):
        tokens = self.canvas.find_withtag("token")
        tokenlist = []
        for token in tokens:
            y = self.canvas.coords(token)[1]/self._y_scale
            x = self.canvas.coords(token)[0]
            tokenlist.append((x, y))
        return tokenlist

    def _create_brush(self, x):
        '''Create a brush at x coordinate'''
        r = self._brush_data['brush']
        f = self._brush_data['feather']
        f_col = self.featherFill
        b_col = self.brushFill
        y0 = self.canvas.canvasy(-10000) #0
        y1 = self.canvas.canvasy(10000) #400
        for index in range(-1, 2):
            x0 = (self._brush_offset*index) + x-r
            x1 = (self._brush_offset*index) + x+r
            feather = self.canvas.create_rectangle(x0-f, y0, x1+f, y1, width=0,
                fill = f_col, tags="feather") 
        for index in range(-1, 2):
            x0 = (self._brush_offset*index) + x-r
            x1 = (self._brush_offset*index) + x+r
            brush = self.canvas.create_rectangle(x0, y0, x1, y1, width=0,
                fill = b_col, tags="brush")
        
    def OnTokenButtonPress(self, event):
        '''Begin drag of an object'''
        # record the item and its location
        self._appendToHistory()
        self._future = []
        x = self.canvas.canvasx(event.x)
        y0 = self.canvas.canvasy(-10000)
        y1 = self.canvas.canvasy(10000)
        r = self._brush_data['brush']
        f = self._brush_data['feather']
        o = self._brush_offset
        for i in range(-1, 2):
            self._drag_data["items"].append(self.canvas.find_overlapping(
                                (o*i)+x-r, y0, (o*i)+x+r, y1))
            self._drag_data["feather"].append(self.canvas.find_overlapping(
                                (o*i)+x-(r+f), y0, (o*i)+x+(r+f), y1))
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
        o = self._brush_offset
        # compute how much this object has moved
        delta_y = (event.y - self._drag_data["y"])*factor
        # move the object the appropriate amount
        for i in range(-1, 2):
            x = (o*i) + self._drag_data['x']
            r = self._brush_data['brush']
            f = self._brush_data['feather']
            f_mode = self._feathermode.get()
            for item in self._drag_data['items'][i+1]:
                if "token" in self.canvas.gettags(item):
                    self.canvas.move(item, 0, delta_y)
                    # move feathered objects
            for item in self._drag_data['feather'][i+1]:
                if "token" in self.canvas.gettags(item) and item not in self._drag_data['items'][i+1]:
                    token_x = self.canvas.coords(item)[0]
                    delta_x = abs(x-token_x)-r
                    feather = tf.feather(f, f_mode, delta_x)
                    self.canvas.move(item, 0, delta_y*feather)               
        # record the new position
        self._drag_data["y"] = event.y
        self.MouseLabelChange(event, mode="y", factor=factor)
        # draw slope
        self._find_slope()

    def ShiftOnTokenMotion(self, event):
        '''Handle shift-dragging of an object'''
        self.OnTokenMotion(event, factor=0.1)

    def _contrastDialog(self):
        dialog = ContrastDialog(self)
        self.wait_window(dialog.top)

    def _contrast(self, weight=0):
        '''Scale all height points'''
        self._appendToHistory()
        self._future = []
        for item in self.canvas.find_withtag('token'):
            x, z = self.canvas.coords(item)[0:2]
            z *= weight
            self.canvas.coords(item, x, z, x, z)
        #draw slope
        self._scrollregion()

    def _roughenDialog(self):
        dialog = RoughenDialog(self)
        self.wait_window(dialog.top)
        
    def _roughen(self, potencia = 4, magnitude = 0.02, variation = 100):
        '''Apply small random elevation changes to all height points'''
        self._appendToHistory()
        self._future = []
        scale = magnitude*self._y_scale
        v = variation
        for item in self.canvas.find_withtag('token'):
            delta_y = ((randint(0, v)/v)**potencia)*scale
            neg = randint(0, 1)
            if neg:
                delta_y*=-neg
            self.canvas.move(item, 0, delta_y)
        #draw slope
        self._scrollregion()

    def _smoothenDialog(self):
        dialog = SmoothenDialog(self)
        self.wait_window(dialog.top)
        
    def _smoothen(self, tol=0.04, minL=6, maxL=100):
        '''Apply a smoothening algorithm to all height points'''
        self._appendToHistory()
        self._future = []
        c = self.canvas
        heightsList = [c.coords(i)[0:2] for i in c.find_withtag('token')]

        rounded = tf.smoothen(heightsList, tol, minL, maxL)
        for item in c.find_withtag('token'):
            x, z = rounded.pop(0)[0:2]
            c.coords(item, x, z, x, z)
        #draw slope
        self._scrollregion()

    def _translateDialog(self):
        dialog = TranslateDialog(self)
        self.wait_window(dialog.top)

    def _translate(self, z=0):
        '''Translate all heightpoints along z axis'''
        self._appendToHistory()
        self._future = []
        for item in self.canvas.find_withtag('token'):
            delta_z = z*self._y_scale
            self.canvas.move(item, 0, delta_z)
        #draw slope
        self._scrollregion()
        

    def brushChange(self, x=None, r=None, f=None):
        if r == None:
            r = self._brush_data['brush']
        if f == None:
            f = self._brush_data['feather']
        if x == None:
            x = self._brush_data['x']
        y0 = self.canvas.canvasy(-10000)
        y1 = self.canvas.canvasy(10000)
        o = self._brush_offset
        brushes = self.canvas.find_withtag("brush")
        feathers = self.canvas.find_withtag("feather")
        index = -1
        for brush in brushes:
            self.canvas.coords(brush, (o*index)+x-r, y0, (o*index)+x+r, y1)
            index += 1
        index = -1
        for feather in feathers:
            self.canvas.coords(feather, (o*index)+x-(r+f), y0, (o*index)+x+(r+f), y1)
            index += 1

    def MouseLabelChange(self, event, mode="all", factor=1):
        canvas = self._brush_data['canvas']
        label = canvas.find_withtag('mouselabel')[0]
        
        if mode == "all":
            x = self.canvas.canvasx(event.x)
            y_coord = y = self.canvas.canvasy(event.y)
        else:
            x = self._mouselabel['x_coord']
            y_coord = self._mouselabel['y_coord']
            y = self.canvas.canvasy(event.y) - y_coord
        r1 = self._brush_data['brush']
        r2 = self._brush_data['feather']

        x_string = '%.3f m\n' %x
        z_string = '%.3f m\n' %(y*factor/self._y_scale)
        b_rad = '%.3f m\n' %r1
        f_rad = '%.3f m' %r2
        
        canvas.itemconfig(label, text = x_string + z_string + b_rad + f_rad)
        #save mouselabel
        self._mouselabel['x_coord'] = x
        self._mouselabel['y_coord'] = y_coord

    def OnMotion(self, event):
        '''Handle moving the brush'''
        self.brushChange(x = self.canvas.canvasx(event.x)) 
        self._brush_data['x'] = self.canvas.canvasx(event.x)
        self.MouseLabelChange(event)
        if self.trackapp:
            self.trackapp.drawTrackIndicator(self.canvas.canvasx(event.x)/self._x_scale)
    def MouseWheel(self, event, mode = 'brush', step=8):
        '''change brush / feather radius'''
        r = self._brush_data[mode]
        if event.delta == -120:
            r -= step
            if r < 0:
                r = 0
        elif event.delta == 120:
            r += step
            if r > 1000:
                r = 1000
        if mode == 'brush':
            self.brushChange(r = r)
        else:
            self.brushChange(f = r)
        self._brush_data[mode] = r
        self.MouseLabelChange(event)

    def ShiftMouseWheel(self, event):
        self.MouseWheel(event, step=1)

    def ControlMouseWheel(self, event, step=8):
        '''change feather radius'''
        self.MouseWheel(event, mode='feather', step=step)

    def ShiftControlMouseWheel(self, event):
        self.ControlMouseWheel(event, step=1)

    def get_drag_data(self):
        '''Select objects'''
        # record the item and its location
        x = self._brush_data['x']
        y0 = self.canvas.canvasy(-10000)
        y1 = self.canvas.canvasy(10000)
        r = self._brush_data['brush']
        o = self._brush_offset
        for i in range(-1, 2):
            self._drag_data["items"]+=(self.canvas.find_overlapping(
                                (o*i)+x-r, y0, (o*i)+x+r, y1))
        temp_list = []
        for item in self._drag_data['items']:
            if "token" in self.canvas.gettags(item):
                #self._drag_data['items'].remove(item)
                temp_list.append(item)

        self._drag_data['items'] = temp_list

    def FButtonPress(self, event):
        '''Flatten section covered by brush'''
        self._appendToHistory()
        self._future = []
        self.get_drag_data()
        
        '''Transformation'''
        try:
            A = self._drag_data['items'].pop(0)
            A = self.canvas.coords(A)
            B = self._drag_data['items'].pop(-1)
            B = self.canvas.coords(B)

            for P in self._drag_data['items']:
                new_xy = tf.flatten(A, B, self.canvas.coords(P))
                self.canvas.coords(P, new_xy)
        except IndexError:
            None

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
        self.get_drag_data()

        '''Transformation'''
        try:
            A = self._drag_data['items'].pop(0)
            A = _append_slope(self, A)
            B = self._drag_data['items'].pop(-1)
            B = _append_slope(self, B)

            for P in self._drag_data['items']:
                new_xy = tf.spline(A, B, self.canvas.coords(P))
                self.canvas.coords(P, new_xy)
        except IndexError:
            None

        self._drag_data['items'] = []
        # draw slope
        self._find_slope()

    def zkey(self, event):
        self.Zoom()

    def altzkey(self, event):
        self.Zoom(zoom=0.5)

    def Zoom(self, zoom=2, axis='z'):
        '''Zoom in and out on the X or Z axis'''
        
        if axis == 'x':
            index = 0
            self._x_scale *= zoom
        else:
            index = 1
            self._y_scale *= zoom
            try:
                self._first_y *= zoom
            except:
                None

        for item in self.canvas.find_withtag('token'):
            coords = self.canvas.coords(item) #get the coords of the item
            coords[index] *= zoom #scale the coordinate specified by the keystroke
            x, z = coords[0], coords[1]
            self.canvas.coords(item, x, z, x, z) #apply the new coordinates

        for item in self.canvas.find_withtag('origin'):
            coords = self.canvas.coords(item)
            coords[1] *= zoom
            coords[3] *= zoom
            self.canvas.coords(item, coords[0], coords[1], coords[2], coords[3])

        self._scrollregion()

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
    root = tk.Tk()
    #token_list = [(i*0.9, 0) for i in range(10, 1000, 7)]
    app = ElevationEditor(root)  
    app.mainloop()
