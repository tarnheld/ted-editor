from skimage import measure
import numpy as np
import struct

import math as m

from PIL import Image

from simplify import simplify

import argparse

parser = argparse.ArgumentParser(description='convert apk heightmaps to floating point tiff')
parser.add_argument('file', type=str, help='the apk heightmap file')

args = parser.parse_args()

hdr=b'\x33\x13\x26\xc3\x33\x13\x26\x43\x02\x00\x20\xc1\x33\x13\xa1\x43'


with open(args.file, mode='rb') as file:
    raw = file.read()
    print(struct.unpack_from("<4xIII",raw,0x1020))
    print(struct.unpack_from("<ffff",raw,0x1030))
    t,w,h = struct.unpack_from("<4xIII",raw,0x1020)
    e1,e2,e3,e4 = struct.unpack_from("<ffff",raw,0x1030)
    dt = np.dtype("half")
    dt = dt.newbyteorder('<')
    img = np.frombuffer(raw,dtype=dt,offset=0x1040,count=w*h)
    print (img.shape)

    img = img.reshape((w,h))

    imin = np.amin(img)
    imax = np.amax(img)

    extents = np.array((e1,e2,e3,e4))
    np.savez_compressed(args.file, extents = extents, heightmap=img)

    fimg = img.astype(np.float32)
    
    fimg.reshape((w*h,1))
    pimg = Image.frombytes('F',(w,h), fimg.tostring(),'raw','F;32NF')

    pimg.save(args.file + ".tif")

    hmin = e1 * (1-imin) + e2 * imin
    hmax = e1 * (1-imax) + e2 * imax

    contours = []
    hstep = 2.5
    nc = m.ceil((hmax-hmin)/hstep) 
    for i in range(nc):
        hgt = imin + i*hstep/(hmax-hmin)
        npc = measure.find_contours(img, hgt)
        cs = []
        for c in npc:
            c = simplify(c,5,True)
            cs.append(c)
        cs = np.array(cs)
        contours.append(cs)
    np.savez_compressed(args.file+"-contours", *contours)

    
    #     mi,ma = float(np.amin(img)),float(np.amax(img))
    #     print("contour",mi,ma)
    #     for i in range(50):
    #       d = float(mi*(1-i/50)+ma*i/50)
    #       print("contour",d)
    #       npc = measure.find_contours(img, d)
    #       for n,c in enumerate(npc):
    #         contours = [((x[1]-512)/1024*3499.99975586*2,(x[0]-512)/1024*3499.99975586*2) for x in c]
    #         if norm(c[-1] - c[0]) < 0.01:
    #           self.canvas.create_polygon(contours,fill="",outline='red',tag="contour")
    #         else:
    #           self.canvas.create_line(contours,fill='green',tag="contour")
    # except FileNotFoundError:
    #   print("file not found!")
    #   return
    # try:
    #   self.img = Image.open(path)
    # except:
    #   try:
    #     with open(path, mode='rb') as file:
    #       raw = file.read()
    #       self.img = Image.frombytes("F",(1024,1024),raw,"raw","F;16")
          
    #       print(self.img.getpixel((4,4)))
    #       f = 1.0 / 2**8
    #       self.img = self.img.point(lambda x: x * f)
    #       print(self.img.getpixel((4,4)))
          
    #       self.img = self.img.resize((8192,8192))
    #       self.img = self.img.filter(ImageFilter.CONTOUR)
    #   except FileNotFoundError:
    #     print("file not found!")
    #     return
      
    # self.ix =2*3499.99975586
    # f = self.ix/2049.0
    # print (f)
    # #self.img = self.img.transform((int(self.ix),int(self.ix)),Image.AFFINE,data=(f,0,0,0,f,0))
    # self.img = self.img.resize((int(self.ix),int(self.ix)))
      
    # self.simg = self.img
    # self.pimg = ImageTk.PhotoImage(self.img)
    # self.imgcid = self.canvas.create_image(-2048, -2048, image=self.pimg, anchor=tk.NW)
