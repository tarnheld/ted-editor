import tkinter as tk
import tkinter.ttk as ttk
from tkinter import font as tkFont
import transformations as tf

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

class ResampleDialog:

    def __init__(self, parent):

        top = self.top = tk.Toplevel(parent)
        self.ElevationEditor = parent

        tk.Label(top, text='Minimum resolution (m)').grid(sticky='w')

        self.e = tk.Entry(top)
        self.e.grid(sticky="we", padx=5)
        self.e.focus()

        b = tk.Button(top, text='OK', command=self.ok)
        b.grid(sticky='we', pady=5)

    def ok(self):
        try:
            minres = float(self.e.get())
        except:
            minres = None
            print('%s is not a valid minimum resolution' %self.e.get())
        if minres!= None:
            self.ElevationEditor._resample(minres)
        self.top.destroy()

class LeftPanel(tk.Frame):

    def __init__(self, parent):
        tk.Frame.__init__(self, parent)

        self.parent = parent
        self.ep_counter = 0

        # brush data display
        l = tk.LabelFrame(self, text = "Brush Data")
        l.grid(padx=5, pady=5, ipadx=2, sticky="we")

        # labels
        for i, item in enumerate(('x coord', 'z coord', 'brush', 'feather', 'mode')):
            tk.Label(l, text=item, pady=0, padx=4, anchor='w').grid(row=i, column=0, sticky='we')

        # coord values
        tk.Label(l, textvariable=parent._mouselabel['x'], anchor='e').grid(row=0, column=1, columnspan=2, sticky='we')
        tk.Label(l, textvariable=parent._mouselabel['z'], anchor='e').grid(row=1, column=1, columnspan=2, sticky='we')
        
        # brush radius
        bradius = FloatSpinbox(l, 0, 1000, width=7, textvariable=parent._brush_data['r'],
                               command=self.callback)
        bradius.grid(row=2, column=1, columnspan=2, sticky='we', padx=1, pady=0)
        bradius.bind('<Return>', self.callback)
        bradius.bind('<FocusOut>', self.callback)

        #feather radius and feather mode
        for i in (0, 1):
            fradius = FloatSpinbox(l, 0, 1000, width=7, textvariable=parent._brush_data['f%d' %i],
                                   command=self.callback)
            fradius.grid(row=3, column=i+1, sticky='we', padx=1, pady=0)
            fradius.bind('<Return>', self.callback)
            fradius.bind('<FocusOut>', self.callback)

            option = ttk.Combobox(l, takefocus=False, width=7,
                                  textvariable = parent._feathermode2[i],
                                  values=('Wave', 'Linear', 'Bowl', 'Plateau'), state="readonly")
            option.set('Wave')
            option.grid(row=4, column=i+1, sticky='we', padx=1, pady=0)

        # create transformations buttons
        l = tk.LabelFrame(self, text = "Transform")
        buttons = [('Contrast', parent._contrastDialog),
                   ('Roughen', parent._roughenDialog),
                   ('Smoothen', parent._smoothenDialog),
                   ('Translate', parent._translateDialog),
                   ('Resample', parent._resampleDialog)]

        for i, button in enumerate(buttons):
            b = tk.Button(l, text=button[0], command = button[1], bd=1)
            b.grid(row=i//2, column=i%2, padx=2, pady=2, sticky="we")
        l.grid(padx=5, pady=5, ipady=2, sticky="we")
        l.columnconfigure(0, weight=1)
        l.columnconfigure(1, weight=1)

        # guides
        self.guideFrame = tk.LabelFrame(self, text = "Elevation profiles")
        self.guideFrame.grid(padx=5, pady=5, ipady=2, sticky='we')
        b = tk.Button(self.guideFrame, text='Import', command = self.import_ep,
                      bd=1)
        b.grid(padx=2, pady=2, sticky='we')
        w = ttk.Separator(self.guideFrame, orient=tk.HORIZONTAL)
        w.grid(padx=2, pady=2, sticky='we')
        self.guideFrame.columnconfigure(0, weight=1)     

    def callback(self, event=None):
        self.focus_set()

    def import_ep(self):
        ID = self.ep_counter
        self.ep_counter +=1
        ep = self.parent._import_ep()
        gp = GuidePanel(self.guideFrame, ID, self.parent, ep)
        gp.grid(padx=2, pady=2, sticky='we')
        self.parent._elevationProfiles.append(gp)

class QuickReference(tk.Toplevel):

    def __init__(self, parent):
        tk.Toplevel.__init__(self, parent)
        self.title('Quick Reference')

        # create instructions panel
        l = tk.LabelFrame(self, text = "Commands")

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
                    'Redo: ctrl + Y')

        for message in messages:
            tk.Label(l, text = message, wraplength=200, anchor='w', justify='left').grid(sticky='we', padx=2)

        l.grid(padx=5, pady=5, ipady=2, sticky="we")
        
class GuidePanel(tk.Frame):
    
    def __init__(self, parent, ID, elevationEditor, elevationProfile=None, bg="#6090CC"):
        tk.Frame.__init__(self, parent, relief='groove', bd=2)
        self.columnconfigure(0, weight=1)

        self.col_bg = bg
        self.labelFont = tkFont.Font(family="Helvetica", size=8)
        self.expanded = False
        self.expFrame = None
        
        self.eleditor = elevationEditor
        self.ID = 'elevationProfile%d' %ID
        self.ep = elevationProfile
        self.filename = self.ep['filename']
        self.data = self.ep['data']
        self._first_x = self.data[0][0]
        self._first_z = self.data[0][1]
        self.outputdata = None

        self.pathfitting = tk.StringVar(value='None')
        self.joint_mode = tk.StringVar(value='None')
        self.x_offset = tk.DoubleVar(value=0)
        self.z_offset = tk.DoubleVar(value=0)
        self._repeat = tk.BooleanVar(value=False)
        self._display = tk.BooleanVar(value=True)
              
        #collapsed UI
        f = tk.Frame(self)
        f.columnconfigure(1, weight=1)
        f.grid(sticky='we')
        
        self.expandButton = IconButton(f, 'folded.gif', command=self.expandToggle)
        self.expandButton.grid(row=0, column=0, padx=2)
        l = tk.Label(f, text=self.title(), font=self.labelFont, anchor='w')
        l.grid(row=0, column=1, sticky='we')
        b = IconButton(f, 'xicon.gif', command=self.close)
        b.grid(row=0, column=2, padx=2)

        sep = ttk.Separator(self, orient=tk.HORIZONTAL)
        sep.grid(padx=2, sticky='we')

        #draw guide
        self.draw()

    def title(self):
        titlelimit = 20
        if len(self.filename) > titlelimit:
            title = self.filename[:titlelimit-2]+'...'
        else:
            title = self.filename
        return title

    def close(self):
        tag = self.ID
        self.eleditor.canvas.delete(tag)
        self.eleditor._elevationProfiles.remove(self)
        self.destroy()

    def expandToggle(self):
        b = self.expandButton
        if self.expanded:
            self.expanded=False
            b.icon = tk.PhotoImage(file='folded.gif')
            self.collapse()
        else:
            self.expanded=True
            b.icon = tk.PhotoImage(file='expanded.gif')
            self.expand()
        b.config(image=self.expandButton.icon)

    def expand(self):
        self.expFrame = f = tk.Frame(self)
        f.columnconfigure(0, weight=1)
        f.grid(sticky='we')

        f1 = tk.Frame(f)
        f1.columnconfigure(1, weight=1)
        f1.grid(sticky='we')

        labels = ['Path-fitting', 'Auto-rejoin', 'Offset x', 'Offset z', 'Repeat']

        for row, text in enumerate(labels):
            l = tk.Label(f1, text=text, font=self.labelFont, anchor='w')
            l.grid(row=row, column = 0, sticky='we')

        PF = ttk.Combobox(f1, takefocus=False, width=6, font=self.labelFont,
                                  textvariable = self.pathfitting,
                                  values=('None', 'Stretch', 'Scale'), state="readonly")
        PF.grid(row=0, column = 1, sticky='we')
        PF.bind("<<ComboboxSelected>>", self.draw)

        JM = ttk.Combobox(f1, takefocus=False, width=6, font=self.labelFont,
                                  textvariable = self.joint_mode,
                                  values=('None', 'Lift', 'Bend'), state="readonly")
        JM.grid(row=1, column = 1, sticky='we')
        JM.bind("<<ComboboxSelected>>", self.draw)

        OS_x = FloatEntry(f1, textvariable = self.x_offset, font=self.labelFont, width=5)
        OS_x.grid(row=2, column = 1, sticky='we')
        OS_x.bind('<Return>', self.draw)
        OS_x.bind('<FocusOut>', self.draw)

        OS_z = FloatEntry(f1, textvariable = self.z_offset, font=self.labelFont, width=5)
        OS_z.grid(row=3, column = 1, sticky='we')
        OS_z.bind('<Return>', self.draw)
        OS_z.bind('<FocusOut>', self.draw)

        RP = tk.Checkbutton(f1, variable=self._repeat, command=self.draw)
        RP.grid(row=4, column = 1, sticky='w')

        sep = ttk.Separator(f, orient=tk.HORIZONTAL)
        sep.grid(padx=2, sticky='we')

        f2 = tk.Frame(f)
        for i in range(0, 3):
            f2.columnconfigure(i, weight=1)
        f2.grid(sticky='we')

        DS = tk.Checkbutton(f2, text='Display', width=5, font=self.labelFont,
                            indicatoron=False, bd=1, variable=self._display,
                            command=self.draw)
        DS.grid(row=0, column=0, sticky='wens', pady=2, padx=1)

        self.SET = tk.Button(f2, text='Set', width=5, font=self.labelFont, bd=1, command=self.applyCommand)
        self.SET.grid(row=0, column=1, sticky='we', pady=2, padx=1)

        self.ADD = tk.Button(f2, text='Add', width=5, font=self.labelFont, bd=1, command = lambda: self.applyCommand(mode='add'))
        self.ADD.grid(row=0, column=2, sticky='we', pady=2, padx=1)   

    def collapse(self):
        self.expFrame.destroy()

    def draw(self, event=None):

        def bend(x, z, last_x, delta_z, feather):
            delta_x = abs(last_x-x)
            if delta_x <= feather:
                z -= delta_z * tf.feather(feather, 'Wave', delta_x)
            return z

        def stretch(tokenlist, tracklen):
            '''scales EP in x dimension'''
            first_x = tokenlist[0][0]
            last_x = tokenlist[-1][0]
            factor = tracklen/(last_x-first_x)
            t = [((x-first_x)*factor, z) for (x, z) in tokenlist]
            return t

        def scale(tokenlist, tracklen):
            '''scales EP in x and z dimension'''
            first_x = tokenlist[0][0]
            last_x = tokenlist[-1][0]
            first_z = tokenlist[0][1]
            factor = tracklen/(last_x-first_x)
            t = [((x-first_x)*factor, first_z+(z-first_z)*factor) for (x, z) in tokenlist]
            return t

        def offset(self, tokenlist):
            '''adds an offset (x and z) to the EP'''
            t = [(x+self.x_offset.get(), z+self.z_offset.get()) for (x, z) in tokenlist]
            return t

        def overlap(self, tokenlist, tracklen):
            '''repeats the EP to fill the length of the track'''
       
            joint_mode = self.joint_mode.get()
            first_xz = tokenlist[0]
            last_xz = tokenlist[-1]
            delta_x = last_xz[0] - first_xz[0]
            delta_z = last_xz[1] - first_xz[1]

            if joint_mode == 'Bend':
                feather = delta_x/2
                last_x = last_xz[0]
                t = [(x, bend(x, z, last_x, delta_z, feather)) for (x, z) in tokenlist]
                tokenlist = t

            if joint_mode != 'Lift':
                delta_z = 0
                
            head = tokenlist[:-1]
            tail = tokenlist[1:]

            #add head
            counter=1
            while tokenlist[0][0] > 0:
                t0 = [(x-delta_x*counter, z-delta_z*counter) for (x, z) in head]
                #t0 = [(x-delta_x*counter, z) for (x, z) in head]
                while tokenlist[0][0] > 0:
                    try:
                        tokenlist.insert(0, t0.pop())
                    except IndexError:
                        break
                counter += 1
                if counter > 1000:
                    print('Exceeded iteration limit')
                    break

            #add tail
            counter=1
            while tokenlist[-1][0] < tracklen:
                t1 = [(x+delta_x*counter, z+delta_z*counter) for (x, z) in tail]
                #t1 = [(x+delta_x*counter, z) for (x, z) in tail]
                while tokenlist[-1][0] < tracklen:
                    try:
                        tokenlist.append(t1.pop(0))
                    except IndexError:
                        break
                counter += 1
                if counter > 1000:
                    print('Exceeded iteration limit')
                    break

            #trim at both ends
            while tokenlist[1][0] < 0:
                trim = tokenlist.pop(0)
            while tokenlist[-2][0] > tracklen:
                trim = tokenlist.pop()

            return tokenlist
                    
        e = self.eleditor

        # delete any instance of the current profile from the canvas
        e.canvas.delete(self.ID)
        self.outputdata = None

        # if display is enabled
        if self._display.get():

            if self.expanded:
                for button in (self.ADD, self.SET):
                    button.config(state='normal')
                
            tracklen = e._high_x
            tags = (self.ID, 'ep')
            tokenlist = [(x, z-e._first_z) for (x, z) in self.data]

            # if a track is loaded
            if len(e._tokens) > 0:
                if self.pathfitting.get() == 'Stretch':
                    tokenlist = stretch(tokenlist, tracklen)
                elif self.pathfitting.get() == 'Scale':
                    tokenlist = scale(tokenlist, tracklen)
                tokenlist = offset(self, tokenlist)
                if self._repeat.get():
                    tokenlist = overlap(self, tokenlist, tracklen)

            else:
                tokenlist = offset(self, tokenlist)

            if self.joint_mode.get() == 'Bend' and e.isLoop:
                delta_x = tokenlist[-1][0]-tokenlist[0][0]
                delta_z = tokenlist[-1][1]-tokenlist[0][1]
                first_x = tokenlist[0][0]
                last_x = tokenlist[-1][0]
    
                feather = delta_x/2

                t = [(x, bend(x, z, last_x, delta_z, feather)) for (x, z) in tokenlist]
                tokenlist = t
                
            self.outputdata = tokenlist
       
            e._create_guide(tokenlist, tags)
            e._order()

        #if display is disabled
        elif self.expanded:
            for button in (self.ADD, self.SET):
                button.config(state='disabled')

    def applyCommand(self, mode='set'):
        e = self.eleditor
        ted_tokens = e._gen_tokenlist()
        ep_tokens = self.outputdata
        ep_start = ep_tokens[0][0]
        ep_end = ep_tokens[-1][0]

        ted_head = [token for token in ted_tokens if token[0] < ep_start]
        ted_body = [token for token in ted_tokens if ep_start <= token[0] <= ep_end]
        ted_tail = [token for token in ted_tokens if ep_end < token[0]]
        
        interpolated = tf.interpolation(ted_body, ep_tokens, mode)

        ted_tokens = ted_head+interpolated+ted_tail

        c = e.canvas
        s = e._y_scale

        for i, token in enumerate(e._tokens):
            x, z = ted_tokens[i]
            c.coords(token, x, z*s, x, z*s)
                
        e._scrollregion()
     
        

class CloseButton(tk.Button):

    def __init__(self, parent, **kwargs):
        tk.Button.__init__(self, parent, **kwargs)
        self.icon = tk.PhotoImage(file='xicon.gif')
        self.config(image=self.icon, bd=0)

class IconButton(tk.Button):

    def __init__(self, parent, icon, **kwargs):
        tk.Button.__init__(self, parent, **kwargs)
        self.icon = tk.PhotoImage(file=icon)
        self.config(image=self.icon, bd=0)

class FloatEntry(tk.Entry):

    def __init__(self, parent, **kwargs):
        vcmd = (parent.register(self._validate),
                '%d', '%i', '%P', '%s', '%S', '%v', '%V', '%W')
        tk.Entry.__init__(self, parent, validate = 'key', validatecommand = vcmd,
                          justify='right', **kwargs)

    def _validate(self, action, index, value_if_allowed,
                  prior_value, text, validation_type, trigger_type, widget_name):
        if action != '1':
            return True
        try:
            float(value_if_allowed)
            return True
        except ValueError:
            return False

class FloatSpinbox(tk.Spinbox):

    def __init__(self, parent, from_, to, **kwargs):
        vcmd = (parent.register(self._validate),
                '%d', '%i', '%P', '%s', '%S', '%v', '%V', '%W')
        tk.Spinbox.__init__(self, parent, from_=from_, to=to, justify='right', validate='key', validatecommand=vcmd, **kwargs)
        self.from_ = from_
        self.to = to

    def _validate(self, action, index, value_if_allowed,
                  prior_value, text, validation_type, trigger_type, widget_name):
        if action != '1':
            return True
        try:
            i = float(value_if_allowed)
            return self.from_ <= i <= self.to
        except ValueError:
            return False

