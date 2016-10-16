import tkinter as tk
import read_ted as ted

import math as m

class App(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.pack()
        self.create_cps()

    def create_cps(self):
        self.canvas = tk.Canvas(self,width=1900,height=1000,scrollregion=(-1000,-1000,1000,1000))
        self.hbar=tk.Scrollbar(self,orient=tk.HORIZONTAL)
        self.hbar.pack(side=tk.BOTTOM,fill=tk.X)
        self.hbar.config(command=self.canvas.xview)
        self.vbar=tk.Scrollbar(self,orient=tk.VERTICAL)
        self.vbar.pack(side=tk.RIGHT,fill=tk.Y)
        self.vbar.config(command=self.canvas.yview)
        self.canvas.config(xscrollcommand=self.hbar.set, yscrollcommand=self.vbar.set)

        self.canvas.create_line(-1000,0,1000,0, fill="red")
        self.canvas.create_line(0,-1000,0,1000, fill="red")
        
        with open("example.ted", mode='rb') as file:
            tedfile = file.read()
            hdr     = ted.ted_data_to_tuple("header", ted.header, tedfile, 0)

            if (hdr.id != b"GT6TED\0\0"):
                print("Error: not a GT6 TED file!")
                return
            print (hdr)
            print(hdr.cp_offset, hdr.cp_count)
            
            cps  = ted.ted_data_to_tuple_list("cp",  ted.cp, tedfile, hdr.cp_offset, hdr.cp_count)
            segs = ted.ted_data_to_tuple_list("bank",ted.segment,tedfile,hdr.bank_offset,hdr.bank_count)
            hs   = ted.ted_data_to_tuple_list("height",ted.height,tedfile,hdr.height_offset,hdr.height_count)

            #print(hdr)
            tl = 0
            for i,cp in enumerate(cps):
                lx = (cps[i-1].x - cp.x)
                ly = (cps[i-1].y - cp.y)
                l = m.sqrt(lx*lx + ly*ly)
                self.canvas.create_text(cp.x,cp.y+10,text="%i"%(i))
                if (ted.SegType(cp.segtype) != ted.SegType.Straight):
                    
                    dx = (cps[i-1].x - cp.center_x)
                    dy = (cps[i-1].y - cp.center_y)
                    dx2 = (cp.x - cp.center_x)
                    dy2 = (cp.y - cp.center_y)
                   
                    r = m.sqrt(dx*dx + dy*dy)

                    
                    ast = m.atan2(-dy, dx)  * 180/m.pi;
                    aen = m.atan2(-dy2,dx2) * 180/m.pi;
                    aex = aen - ast


                    if (ted.SegType(cp.segtype) == ted.SegType.Arc1CCW or
                        ted.SegType(cp.segtype) == ted.SegType.Arc2CCW):
                        if (aex < 0): aex += 360
                    if (ted.SegType(cp.segtype) == ted.SegType.Arc1CW or
                        ted.SegType(cp.segtype) == ted.SegType.Arc2CW):
                        if (aex > 0): aex -= 360

                    sl = (abs(aex) / 180 * m.pi) * r

                    
                    
                    self.canvas.create_arc([cp.center_x-r,cp.center_y+r,cp.center_x+r,cp.center_y-r],start=ast,extent=aex,style=tk.ARC,outline="blue")
                    self.canvas.create_oval([cp.center_x-5,cp.center_y-5,cp.center_x+5,cp.center_y+5],fill="blue") # center of arc
                    self.canvas.create_text(cp.center_x,cp.center_y+10*i,text="%i"%(i))
                else:
                    sl = l
                    self.canvas.create_line(cp.x,cp.y,cps[i-1].x,cps[i-1].y,fill="green")
                s = segs[i-1]
                z = hs[s.total_divisions+s.divisions].height - hs[s.total_divisions].height
                if i == 0: z = 0.0
                sl = m.sqrt(sl*sl+z*z) 
                tl += sl
                div = segs[i-1].divisions
                print(i,ted.SegType(cp.segtype),sl/div, sl, tl )
                print(i-1,segs[i-1].vlen/segs[i-1].divisions,segs[i-1].vstart,segs[i-1].vlen)

        print ("track length = ",tl)
        self.canvas.config(scrollregion=self.canvas.bbox(tk.ALL))
        self.canvas.pack(side=tk.LEFT,expand=True,fill=tk.BOTH)

app = App()

app.master.title("The CP Viewer")
app.master.maxsize(1900,1000)

app.mainloop()
