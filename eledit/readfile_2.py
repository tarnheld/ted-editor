import os.path
from ntpath import basename
import math
import struct
import tkinter as tk
from tkinter import filedialog
from time import strftime

class TrackObject():
    def __init__(self, file, filename=''):
        self.file = file
        self.filename = filename

        #extract the data
        self.header = self.extract_header(self.file)
        
        self.cps = self.extract_cps(self.file, self.header)
        self.banks = self.extract_banks(self.file, self.header)
        self.heights = self.extract_heights(self.file, self.header)
        self.checkpoints = self.extract_checkpoints(self.file, self.header)
        self.roads = self.extract_roads(self.file, self.header)
        self.decorations = self.extract_decorations(self.file, self.header)

        self.measure_track()

    def measure_track(self):
        self.mod = self.heights.copy()
        
        self.banklengths2d = []
        self.banklengths3d = []
        self.tracklength2d = 0
        self.tracklength3d = 0
        self.measure_track_length()

        self.checkpoints2d = [self.get_distance(checkpoint, 0) for checkpoint in self.checkpoints]
        self.finishline2d = self.get_distance(self.header['finishline'][-1], 0)
        self.startline2d = self.get_distance(self.header['startline'][-1], 0)
        for item in self.roads+self.decorations:
            item['vposExcludeHeight'] = self.get_distance(item['vposIncludeHeight'], 0)
            item['vposExcludeHeight2'] = self.get_distance(item['vposIncludeHeight2'], 0)
        self.tokens = self.make_tokens()

    def get_distance(self, distance, mode=0):
        '''Modes: 0 = vpos to vpos2d, 1 = vpos2d to vpos3d, 2 = vpos3d to vpos2d'''

        def get_bank_index(self, distance, mode=0):
            if mode == 0:       #convert from vpos to vpos2d
                banklengths = [bank['vpos'] for bank in self.banks]
            elif mode == 1:     #convert from vpos2d to vpos3d
                banklengths = [bank['vpos2d'] for bank in self.banks]
            else:               #convert from vpos3d to vpos2d
                banklengths = [bank['vpos3d'] for bank in self.banks]
            bank_index = -1
            for length in banklengths:
                if distance >= length:
                    bank_index += 1
                else:
                    break
            return bank_index

        def get_dist_fraction(self, distance, mode=0):
            bank_index = get_bank_index(self, distance, mode)
            bank = self.banks[bank_index]
            if mode == 0:       #from vpos to vpos2d
                bank_start = bank['vpos']
                vlen = bank['vlen']
            elif mode == 1:     #from vpos2d to vpos3d
                bank_start = bank['vpos2d']
                vlen = bank['vlen2d']
            else:               #from vpos3d to vpos2d
                bank_start = bank['vpos3d']
                vlen = bank['vlen3d']
                
            fraction = (distance - bank_start)/vlen

            return fraction, bank_index

        if mode == 1:   #from vpos2d to vpos3d
            bl = self.banklengths3d
        else:           #from vpos or vpos3d to vpos2d
            bl = self.banklengths2d
        
        fraction, bank_index = get_dist_fraction(self, distance, mode)
        distance = sum(bl[:bank_index]) + bl[bank_index]*fraction
        return distance
        
    def extract_header(self, file):
        headerDict = {'id': [0, 8, '>Q'],   #'key': [byteposition, bytesize, '>byteformat']
                      'version': [8, 4, '>l'],
                      'sceneryindex': [12, 4, '>L'],
                      'roadwidth': [16, 4, '>f'],
                      'm_trackwidth_a': [20, 4, '>f'],
                      'm_trackwidth_b': [24, 4, '>f'],
                      'tracklength': [28, 4, '>f'],
                      'datetime': [32, 4, '>L'],
                      'isloopcourse': [36, 4, '>L'],
                      'byte padding0': [40, 8, '>Q'],
                      'homestraightlength': [48, 4, '>f'],
                      'elevationdifference': [52, 4, '>f'],
                      'cournercount': [56, 4, '>l'],
                      'finishline': [60, 4, '>f'],
                      'startline': [64, 4, '>f'],
                      'byte padding1': [68, 8, '>Q'],
                      'cps_offset': [76, 4, '>L'],
                      'cps_entry_count': [80, 4, '>l'],
                      'reserved1_offset': [84, 4, '>L'],
                      'reserved1_entry_count': [88, 4, '>l'],
                      'reserved2_offset': [92, 4, '>L'],
                      'reserved2_entry_count': [96, 4, '>l'],
                      'reserved3_offset': [100, 4, '>L'],
                      'reserved3_entry_count': [104, 4, '>l'],
                      'banks_offset': [108, 4, '>L'],
                      'banks_entry_count': [112, 4, '>l'],
                      'heights_offset': [116, 4, '>L'],
                      'heights_entry_count': [120, 4, '>l'],
                      'checkpoints_offset': [124, 4, '>L'],
                      'checkpoints_entry_count': [128, 4, '>l'],
                      'roads_offset': [132, 4, '>L'],
                      'roads_entry_count': [136, 4, '>l'],
                      'decorations_offset': [140, 4, '>L'],
                      'decorations_entry_count': [144, 4, '>l'],
                      'byte padding2': [148, 8, '>Q']}
        
        for key in headerDict:
            v = headerDict[key]
            fmt = v[2]
            start = v[0]
            end = start + v[1]
            value = struct.unpack(fmt, file[start:end])[0]
            v.append(value)
        return headerDict

    def extract_cps(self, file, header):
        b = 20 #length in bytes
        offset = header['cps_offset'][-1]
        entry_count = header['cps_entry_count'][-1]
        end = offset + (entry_count*b)
        CP_list = []
        for i in range(offset, end, b):
            d = {}
            d['formtype'] = struct.unpack('>L', file[i:i+4]) [0]
            d['x'] = struct.unpack('>f', file[i+4:i+8]) [0]
            d['y'] = struct.unpack('>f', file[i+8:i+12]) [0]
            d['x2'] = struct.unpack('>f', file[i+12:i+16]) [0]
            d['y2'] = struct.unpack('>f', file[i+16:i+20]) [0]
            CP_list.append(d)
        return CP_list

    def extract_banks(self, file, header):
        b = 28
        offset = header['banks_offset'][-1]
        entry_count = header['banks_entry_count'][-1]
        end = offset + (entry_count*b)
        banklist = []
        for i in range(offset, end, b):
            d = {}
            d['m_bank'] = struct.unpack('>f', file[i:i+4]) [0]
            d['m_shiftPrev'] = struct.unpack('>f', file[i+4:i+8]) [0]
            d['m_shiftNext'] = struct.unpack('>f', file[i+8:i+12]) [0]
            d['divNum'] = struct.unpack('>L', file[i+12:i+16]) [0]
            d['unk'] = struct.unpack('>L', file[i+16:i+20]) [0]
            d['vpos'] = struct.unpack('>f', file[i+20:i+24]) [0]
            d['vlen'] = struct.unpack('>f', file[i+24:i+28]) [0]         
            banklist.append(d)
        return banklist

    def extract_heights(self, file, header):
        b = 4
        offset = header['heights_offset'][-1]
        entry_count = header['heights_entry_count'][-1]
        end = offset + (entry_count*b)
        heights = self.extract_floats(file, offset, end)
        return heights

    def extract_checkpoints(self, file, header):
        b = 4
        offset = header['checkpoints_offset'][-1]
        entry_count = header['checkpoints_entry_count'][-1]
        end = offset + (entry_count*b)
        checkpoints = self.extract_floats(file, offset, end)
        return checkpoints

    def extract_floats(self, file, offset, end):
        floatlist = []
        for i in range(offset, end, 4):
            _float = struct.unpack('>f', file[i:i+4])[0]
            floatlist.append(_float)
        return floatlist

    def extract_roads(self, file, header):
        b = 20
        offset = header['roads_offset'][-1]
        entry_count = header['roads_entry_count'][-1]
        end = offset + (entry_count*b)
        roadlist = []
        for i in range(offset, end, b):
            d = {}
            d['uuid'] = struct.unpack('>Q', file[i:i+8]) [0]
            d['flag'] = struct.unpack('>l', file[i+8:i+12]) [0]
            d['vposIncludeHeight'] = struct.unpack('>f', file[i+12:i+16]) [0]
            d['vposIncludeHeight2'] = struct.unpack('>f', file[i+16:i+20]) [0]
            roadlist.append(d)
        return roadlist

    def extract_decorations(self, file, header):
        b = 24
        offset = header['decorations_offset'][-1]
        entry_count = header['decorations_entry_count'][-1]
        end = offset + (entry_count*b)
        decorlist = []
        for i in range(offset, end, b):
            d = {}
            d['m_arr_Cliff'] = struct.unpack('>Q', file[i:i+8]) [0]
            d['railtype'] = struct.unpack('>l', file[i+8:i+12]) [0]
            d['vposIncludeHeight'] = struct.unpack('>f', file[i+12:i+16]) [0]
            d['vposIncludeHeight2'] = struct.unpack('>f', file[i+16:i+20]) [0]
            d['tracktype'] = struct.unpack('>l', file[i+20:i+24]) [0]
            decorlist.append(d)
        return decorlist

    def make_tokens(self):
        '''Returns a list of token tuples (x, y) from an extracted TED file'''

        def get_bank_index(self, index):
            bank_index = -1
            for bank in self.banks:
                if index >= bank['unk']:
                    bank_index += 1
                else:
                    break
            return bank_index

        def get_distance(self, index):
            bank_index = get_bank_index(self, index)
            
            bank = self.banks[bank_index]
            #banklen = self.banklengths2d[bank_index]
            
            divNum = bank['divNum']
            unk = bank['unk']
            vpos = bank['vpos2d']
            vlen = bank['vlen2d']
            distance = vlen/divNum*(index-unk)+vpos
            return distance

        tokens = [(get_distance(self, i), z) for i, z in enumerate(self.mod)]

        return tokens

    def measure_track_length(self):
        '''measures track length excluding height'''

        heights = self.mod.copy()
              
        self.tracklength2d = 0
        self.tracklength3d = 0
        self.banklengths2d = []
        self.banklengths3d = []
        
        prev = None
        for index, cp in enumerate(self.cps):
            this = [cp['x'], cp['y'], heights[0]] #x, y, z
            if index == 0:    #if first
                dist2d = dist3d = 0
            else:               #if not first
                if cp['formtype'] in (0, 3): #if straight
                    x = this[0]-prev[0]
                    y = this[1]-prev[1]
                    dist2d = (x**2+y**2)**0.5
                else: #if arc
                    center = (cp['x2'], cp['y2'])
                    x = this[0]-center[0]
                    y = this[1]-center[1]
                    radius = hyp = (x**2+y**2)**0.5

                    x = this[0]-prev[0]
                    y = this[1]-prev[1]
                    opp = ((x**2+y**2)**0.5)/2

                    angle = math.asin(opp/hyp)*2
                    arcLen = dist2d = angle*radius

                z = this[2]-prev[2]
                dist3d = (dist2d**2+z**2)**0.5
                
                bank = self.banks[index-1]
                bank['vpos2d'] = self.tracklength2d
                bank['vpos3d'] = self.tracklength3d
                bank['vlen2d'] = dist2d
                bank['vlen3d'] = dist3d

                self.tracklength2d += dist2d
                self.tracklength3d += dist3d
                self.banklengths2d.append(dist2d)
                self.banklengths3d.append(dist3d)

            try:
                bank = self.banks[index]
                heights = heights[bank['divNum']:]
            except IndexError:
                None
            prev = this

    def update_data(self):
        
        def update_header(self):
            self.header['tracklength'][-1] = sum([bank['vlen'] for bank in self.banks])
            self.header['elevationdifference'][-1] = max(self.mod)-min(self.mod)
            self.header['finishline'][-1] = self.get_distance(self.finishline2d, 1)
            self.header['startline'][-1] = self.get_distance(self.startline2d, 1)

            #update offsets and entry counts:
            cps_offset = sum(self.header[item][1] for item in self.header)
            cps_entry_count = len(self.cps)
            reserveds_offset = cps_offset+cps_entry_count*20
            banks_offset = reserveds_offset
            banks_entry_count = len(self.banks)
            heights_offset = banks_offset+banks_entry_count*28
            heights_entry_count = len(self.mod)
            checkpoints_offset = heights_offset+heights_entry_count*4
            checkpoints_entry_count = len(self.checkpoints)
            roads_offset = checkpoints_offset+checkpoints_entry_count*4
            roads_entry_count = len(self.roads)
            decorations_offset = roads_offset+roads_entry_count*20
            decorations_entry_count = len(self.decorations)
            
            self.header['cps_offset'][-1] = cps_offset
            self.header['cps_entry_count'][-1] = cps_entry_count
            self.header['reserved1_offset'][-1] = reserveds_offset
            self.header['reserved1_entry_count'][-1] = 0
            self.header['reserved2_offset'][-1] = reserveds_offset
            self.header['reserved2_entry_count'][-1] = 0
            self.header['reserved3_offset'][-1] = reserveds_offset
            self.header['reserved3_entry_count'][-1] = 0
            self.header['banks_offset'][-1] = banks_offset
            self.header['banks_entry_count'][-1] = banks_entry_count
            self.header['heights_offset'][-1] = heights_offset
            self.header['heights_entry_count'][-1] = heights_entry_count
            self.header['checkpoints_offset'][-1] = checkpoints_offset
            self.header['checkpoints_entry_count'][-1] = checkpoints_entry_count
            self.header['roads_offset'][-1] = roads_offset
            self.header['roads_entry_count'][-1] = roads_entry_count
            self.header['decorations_offset'][-1] = decorations_offset
            self.header['decorations_entry_count'][-1] = decorations_entry_count

        def update_banks(self):
            for bank in self.banks:
                bank['vpos'] = bank['vpos3d']
                bank['vlen'] = bank['vlen3d']

        def update_checkpoints(self):
            self.checkpoints = [self.get_distance(c, 1) for c in self.checkpoints2d]

        def update_roadsAndDeco(self, items):
            for item in items:
                item['vposIncludeHeight'] = self.get_distance(item['vposExcludeHeight'], 1)
                item['vposIncludeHeight2'] = self.get_distance(item['vposExcludeHeight2'], 1)

        self.measure_track_length()
        
        update_banks(self)
        update_checkpoints(self)
        update_roadsAndDeco(self, self.roads)
        update_roadsAndDeco(self, self.decorations)
        update_header(self) 

def initFromTedFile(tedfile, filename = ''):
    '''Returns a TrackObject from a TED file'''
    Track = TrackObject(tedfile, filename)
    return Track

def load_TED():
    '''Loads a TED file and returns a TrackObject'''
    root=tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    file_opt = options = {}
    options['defaultextension'] = '.ted'
    options['filetypes'] = [('GT6TED', '.ted')]
    options['initialdir'] = 'TED'
    options['parent'] = root
    options['title'] = 'Open file'
    
    path = filedialog.askopenfilename(**file_opt)
    filename = basename(path)
    root.destroy()
    try:
        with open(path, mode='rb') as file:
            tedfile = file.read()
            Track = initFromTedFile(tedfile, filename)
            
        return Track
    except FileNotFoundError:
        return None

def generateTedFile(Track):
    '''Generates a TED file from a TrackObject'''
    Track.update_data()
    
    #pack header
    buf = bytearray(156)
    header = Track.header
    for key in header:
        item = header[key]
        fmt = item[-2]
        offset = item[0]
        val = item[-1]
        struct.pack_into(fmt, buf, offset, val)

    #pack cps
    cps = Track.cps
    for cp in cps:
        buf += struct.pack('>L', cp['formtype'])
        buf += struct.pack('>f', cp['x'])
        buf += struct.pack('>f', cp['y'])
        buf += struct.pack('>f', cp['x2'])
        buf += struct.pack('>f', cp['y2'])      

    #pack banks
    banks = Track.banks
    for bank in banks:
        buf += struct.pack('>f', bank['m_bank'])
        buf += struct.pack('>f', bank['m_shiftPrev'])
        buf += struct.pack('>f', bank['m_shiftNext'])
        buf += struct.pack('>L', bank['divNum'])
        buf += struct.pack('>L', bank['unk'])
        buf += struct.pack('>f', bank['vpos'])
        buf += struct.pack('>f', bank['vlen']) 

    #pack heights
    heights = Track.mod
    for height in heights:
        buf += struct.pack('>f', height)

    #pack checkpoints
    checkpoints = Track.checkpoints
    for checkpoint in checkpoints:
        buf += struct.pack('>f', checkpoint)

    #pack roads
    roads = Track.roads
    for road in roads:
        buf += struct.pack('>Q', road['uuid'])
        buf += struct.pack('>l', road['flag'])
        buf += struct.pack('>f', road['vposIncludeHeight'])
        buf += struct.pack('>f', road['vposIncludeHeight2'])
        
    #pack decorations
    decorations = Track.decorations
    for decoration in decorations:
        buf += struct.pack('>Q', decoration['m_arr_Cliff'])
        buf += struct.pack('>l', decoration['railtype'])
        buf += struct.pack('>f', decoration['vposIncludeHeight'])
        buf += struct.pack('>f', decoration['vposIncludeHeight2'])
        buf += struct.pack('>l', decoration['tracktype'])

    tedfile = buf
    return tedfile

def export_TED(Track):
    '''Exports a TrackObject as a TED file'''
    tedfile = generateTedFile(Track)

    #savedialog
    root=tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    file_opt = options = {}
    options['defaultextension'] = '.ted'
    options['filetypes'] = [('GT6TED', '.ted')]
    options['initialdir'] = 'Output'
    if Track.filename == '' or Track.filename[-4:] != '.ted':
        filename = "%s.ted" % strftime("%Y%m%d_%H%M%S")
    else:
        filename = Track.filename
    options['initialfile'] = filename
    options['parent'] = root
    options['title'] = 'Save file'
    filename = filedialog.asksaveasfilename(**options)
    root.destroy()

    #save the file
    try:
        with open(filename, 'wb') as file:
            file.write(tedfile)
    except FileNotFoundError:
        print("FileNotFoundError: No such file or directory.")

def comp(A, B):
    '''compares two files to see if they're identical'''
    for index, byte in enumerate(A):
        if byte != B[index]:
            print(index, byte, B[index])
            break
    print('done')
            
    
