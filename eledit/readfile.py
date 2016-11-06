import os.path
import struct
import tkinter as tk
from tkinter import filedialog
from time import strftime

def load_struct():
    '''Loads and extracts a TED structure from file''' 

    def extract_structure(file):
        '''Extracts head, banks, heights and tail from a loaded TED file'''

        def extract_heightlist(file, offset, end):
            heightlist = []
            for i in range(offset, end, 4):
                height = struct.unpack('>f', file[i:i+4])[0]
                heightlist.append(height)
            return heightlist

        def extract_banklist(file, offset, end):
            banklist = []
            for i in range(offset, end, 28):
                banking = struct.unpack('>f', file[i:i+4]) [0]
                divNum = struct.unpack('>L', file[i+12:i+16]) [0]
                unk = struct.unpack('>L', file[i+16:i+20]) [0]
                vpos = struct.unpack('>f', file[i+20:i+24]) [0]
                vlen = struct.unpack('>f', file[i+24:i+28]) [0]
                
                banklist.append({'banking': banking, 'vlen':vlen,
                                  'divNum':divNum, 'unk':unk, 'vpos':vpos})
            return banklist

        def make_tokens(structure):
            '''Creates token data (x, y) from an extracted TED structure'''

            def get_bank_index(index, banks):
                bank_index = -1
                for bank in banks:
                    if index >= bank['unk']:
                        bank_index += 1
                    else:
                        break
                return bank_index

            def get_distance(bank, index):
                vlen = bank['vlen']
                divNum = bank['divNum']
                unk = bank['unk']
                vpos = bank['vpos']
                distance = vlen/divNum*(index-unk)+vpos
                return distance

            heights = structure['mod']
            banks = structure['banks']

            index = 0
            for height in heights:
                bank_index = get_bank_index(index, banks)
                bank = banks[bank_index]
                x = get_distance(bank, index)
                structure['tokens'].append((x, height))
                index += 1

            return structure

        bank_offset = struct.unpack('>L', file[108:112])[0]
        head_end = bank_end = height_offset = struct.unpack('>L', file[116:120])[0]
        height_end = tail_offset = struct.unpack('>L', file[124:128])[0]

        head = file[:head_end]
        tail = file[tail_offset:]
        
        heightlist = extract_heightlist(file, height_offset, height_end)
        banklist = extract_banklist(file, bank_offset, bank_end)
        mod = list(heightlist) #creates copy for modding
        
        structure = {'head': head, 'banks': banklist, 'heights': heightlist,
                'mod': mod, 'tail': tail, 'tokens' : []}

        structure = make_tokens(structure)
        return structure  

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
    root.destroy()
    try:
        with open(path, mode='rb') as file:
            fileContent = file.read()
        structure = extract_structure(fileContent)
        return structure
    except FileNotFoundError:
        return None

def export_struct(structure, name='output'):
    '''Exports a TED structure to file'''
    buf = structure['head']
    buf += struct.pack('>%sf' % len(structure['mod']), *structure['mod'])
    buf += structure['tail']
    
    root=tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    file_opt = options = {}
    options['defaultextension'] = '.ted'
    options['filetypes'] = [('GT6TED', '.ted')]
    options['initialdir'] = 'Output'
    options['initialfile'] = "%s.ted" % strftime("%Y%m%d_%H%M%S")
    options['parent'] = root
    options['title'] = 'Save file'
    filename = filedialog.asksaveasfilename(**options)
    root.destroy()
    #filename = os.path.join('Output', name+'.ted')
    try:
        with open(filename, 'wb') as file:
            file.write(buf)
    except FileNotFoundError:
        print("FileNotFoundError: No such file or directory.")
    
