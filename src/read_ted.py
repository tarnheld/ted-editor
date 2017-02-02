import struct
from collections import namedtuple
from enum import Enum

class Scenery(Enum):
    DeathValley = 1
    Eifel       = 2
    Andalusia   = 3
    EifelFlat   = 5
class SegType(Enum):
    Straight = 0
    Arc1CCW  = 1
    Arc2CCW  = 2
    NearlyStraight = 3
    Arc1CW   = 2147483649 # 2^31 + 1 ???
    Arc2CW   = 2147483650 # 2^31 + 2 ???
class RailType(Enum):
    Grass = 0
    Tree = 1
    TreeGroup = 2
    Flower = 3
    Cactus = 4
    CactusGroup = 5
    Curb = 6
    House = 7
    Rock = 8
    Cliff = 9
    Pole = 10
    StreetLamp = 11
    CircuitGate = 12
    Stand = 13
    MarshalPost = 14
    DiamondVision = 15
    Banner = 16
    AutoCliff = 17
    RallyGate = 18
    Audience = 19
    Billboard = 20
    Tent = 21
    CircuitGateOneSide = 22
    RallyGateOneSide = 23
    Limit = 24
class TrackType(Enum):
    LeftA = 0
    LeftB = 1
    LeftAB = 2
    RightA = 3
    RightB = 4
    RightAB = 5
    LeftCurb= 6
    RightCurb = 7
    Limit = 8
#class RailType(Enum):
#    Normal = 0
#    Start = 1
#    Goal = 2
#    CheckPoint = 3
#    HomeStraight = 4
class Side(Enum):
    Right = 0
    Left  = 1
    Start = 2
class Track(Enum):
    Null = 0
    Curb = 1
    A    = 2
    B    = 3
    AB   = 4
    
header="""
8s:id
i:version
i:scenery
f:road_width
f:track_width_a
f:track_width_b
f:track_length
I:datetime
i:is_loop
8x:
f:home_straight_length
f:elevation_diff
i:num_corners
f:finish_line
f:start_line
8x:
I:cp_offset
i:cp_count
I:empty1_offset
i:empty1_count
I:empty2_offset
i:empty2_count
I:empty3_offset
i:empty3_count
I:bank_offset
i:bank_count
I:height_offset
i:height_count
I:checkpoint_offset
i:checkpoint_count
I:road_offset
i:road_count
I:decoration_offset
i:decoration_count
8x:
"""
cp="""
I:segtype
f:x
f:y
f:center_x
f:center_y
"""
segment="""
f:banking
f:transition_prev_vlen
f:transition_next_vlen
I:divisions
i:total_divisions
f:vstart
f:vlen
"""
height="""
f:height
"""
checkpoint="""
f:vpos3d
"""

#8s:uuid
#i:flag 

road="""
Q:uuid
i:side
f:vstart3d
f:vend3d
"""
decoration="""
Q:uuid
I:side
f:vstart3d
f:vend3d
I:tracktype
"""

def fmt_to_spec_and_fields(fmt):
    spec=""
    fields=""
    for line in fmt.splitlines():
        for token in line.split():
            s,f = token.split(":")
            spec   += s
            fields += f + " "
        
    return spec,fields

def ted_data_size(fmt):
    spec, members = fmt_to_spec_and_fields(fmt)
    return struct.calcsize(spec)
def ted_data_tuple(name,fmt):
    spec, members = fmt_to_spec_and_fields(fmt)
    nt = namedtuple(name,members)
    nt.__new__.__defaults__ = (None,) * len(nt._fields)
    return nt
def ted_data_to_tuple(name,fmt,data,offset):
    spec, members = fmt_to_spec_and_fields(fmt)
    spec = ">" + spec
    return namedtuple(name,members)._make(struct.unpack_from(spec, data, offset))
def ted_data_to_tuple_list(name,fmt,data,offset,count):
    spec, members = fmt_to_spec_and_fields(fmt)
    spec = ">" + spec
    sz = struct.calcsize(spec)
    lst = []
    for c in range(count):
        t = namedtuple(name,members)._make(struct.unpack_from(spec, data, offset + c*sz))
        lst.append(t)
    return lst
def tuple_to_ted_data(tpl,fmt,data,offset):
    spec, members = fmt_to_spec_and_fields(fmt)
    spec = ">" + spec
    struct.pack_into(spec,data,offset,*tpl._asdict().values())
    return offset + struct.calcsize(spec)
def tuple_list_to_ted_data(lst,fmt,data,offset,count):
    spec, members = fmt_to_spec_and_fields(fmt)
    spec = ">" + spec
    sz = struct.calcsize(spec)
    for c in range(count):
        struct.pack_into(spec,data,offset + c*sz,*lst[c]._asdict().values())
    return offset + count*sz


if __name__ == "__main__":
    import math as m
    import xml.etree.ElementTree as ET
    tree = ET.parse('rd/andalusia.raildef')
    root = tree.getroot()
    uuids = {}
    names = { "Unknown" : [] }
    railunit = {}
    for ru in root.iter("RailUnit"):
        uuid = int(ru.find("uuid").text)
        railunit[uuid] = ru
        uuids[uuid]    = "Unknown"
        names["Unknown"].append(uuid)
        
    for rgi in root.iter("RailGroupItem"):
        uuid = int(rgi.find("uuid").text)
        name =     rgi.find("name").text
        uuids[uuid] = name
        names[name] = uuid
        if uuid in names["Unknown"]:
            names["Unknown"].remove(uuid)
    

    
    import argparse

    parser = argparse.ArgumentParser(description='read ted files')
    parser.add_argument('file', type=str, help='the apk heightmap file')

    args = parser.parse_args()

    tedfile = None
    with open(args.file, mode='rb') as file:
        tedfile = file.read()
        
        print("tedfile contains ",len(tedfile),"bytes")
        hdr     = ted_data_to_tuple("header",header,tedfile,0)
        cps     = ted_data_to_tuple_list("cp",cp,tedfile,hdr.cp_offset,hdr.cp_count)
        banks   = ted_data_to_tuple_list("bank",segment,tedfile,hdr.bank_offset,hdr.bank_count)
        heights = ted_data_to_tuple_list("height",height,tedfile,hdr.height_offset,hdr.height_count)
        checkps = ted_data_to_tuple_list("checkpoints",checkpoint,tedfile,hdr.checkpoint_offset,hdr.checkpoint_count)
        troad   = ted_data_to_tuple_list("road",road,tedfile,hdr.road_offset,hdr.road_count)
        deco    = ted_data_to_tuple_list("decoration",decoration,tedfile,hdr.decoration_offset,hdr.decoration_count)

        cpo = ted_data_size(header)
        bo = cpo + ted_data_size(cp)*hdr.cp_count
        ho = bo  + ted_data_size(segment)*hdr.bank_count
        co = ho  + ted_data_size(height)*hdr.height_count
        ro = co  + ted_data_size(checkpoint)*hdr.checkpoint_count
        do = ro  + ted_data_size(road)*hdr.road_count
        end = do + ted_data_size(decoration)*hdr.decoration_count
        print(hdr.cp_offset,"vs",cpo)
        print(hdr.bank_offset,"vs",bo)
        print(hdr.height_offset,"vs",ho)
        print(hdr.checkpoint_offset,"vs",co)
        print(hdr.road_offset,"vs",ro)
        print(hdr.decoration_offset,"vs",do)
        print(len(tedfile),"vs",end)
        #quit()
        print(hdr)
        print("checkpoints")
        radii = []
        for i,x in enumerate(cps):
            print(i,x)
            radii.append(0 if x.segtype == 0 else ((x.center_x-x.x)**2+(x.center_y-x.y)**2)**(1/2))
            print(i,"radius:",radii[i])
        print("banks")
        nh = 0;
        for i,x in enumerate(banks):
            dl = x.vlen/x.divisions
            print(i,x)
            print(i,"   ",dl,x.transition_prev_vlen/dl,x.transition_next_vlen/dl,(x.transition_next_vlen+x.transition_prev_vlen)/dl)
            nh += x.divisions
        print("numheights:",nh)
        print("height")
        for i,x in enumerate(heights):
            print(i,x)
        print("checkpoints")
        for i,x in enumerate(checkps):
            print(i,x)
        print("road")
        unlen = 0.0
        for i,x in enumerate(troad):
            unlen += float(railunit[x.uuid].find("unitLength").text)
            print(i,x, uuids[x.uuid], x.vend3d - x.vstart3d,railunit[x.uuid].find("unitLength").text)
        exlen = (unlen - hdr.track_length)/len(troad)
        print (unlen - hdr.track_length, (unlen - hdr.track_length)/len(troad))
        for i,x in enumerate(troad):
            print (x.vend3d - x.vstart3d - float(railunit[x.uuid].find("unitLength").text) + exlen)
        print("deco")
        for i,x in enumerate(deco):
            if x.uuid in railunit:
                print(i,x,uuids[x.uuid], x.vend3d - x.vstart3d,railunit[x.uuid].find("unitLength").text, RailType(int(railunit[x.uuid].find("decoratedRailType").text)), TrackType(x.tracktype).name)
            else:
                print(i,x, x.vend3d - x.vstart3d, Side(x.side).name,TrackType(x.tracktype).name)
        
        
