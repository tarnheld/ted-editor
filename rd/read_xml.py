import xml.etree.ElementTree as ET
tree = ET.parse('andalusia.raildef')
root = tree.getroot()
from enum import Enum
class RailType(Enum):
    Road = 0
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

uuids = {}
names = { "Unknown" : [] }
railtypes = {}
railunit = {}
for ru in root.iter("RailUnit"):
    uuid = int(ru.find("uuid").text)
    railunit[uuid] = ru
    uuids[uuid]    = "Unknown"
    names["Unknown"].append(uuid)
    rt = int(ru.find("decoratedRailType").text)
    if rt in railtypes:
        railtypes[rt].append(uuid)
    else:
        railtypes[rt] = [uuid]

for rgi in root.iter("RailGroupItem"):
    uuid = int(rgi.find("uuid").text)
    name =     rgi.find("name").text
    uuids[uuid] = name
    names[name] = uuid
    if uuid in names["Unknown"]:
        names["Unknown"].remove(uuid)

print("railtypes")
for rt,uuid in railtypes.items():
    print(rt,RailType(rt),uuid)
    ruchilds = {}
    if rt == 0: continue
    for u in uuid:
        print(uuids[u])
        ru = railunit[u]
        for child in ru:
            if child.tag in ruchilds:
                if child.text.isspace():
                    ruchilds[child.tag].append([c.text for c in child])
                else:
                    ruchilds[child.tag].append(child.text)
            else:
                if child.text.isspace():
                    ruchilds[child.tag] = [[c.text for c in child]]
                else:
                    ruchilds[child.tag] = [child.text]
    for x,t in ruchilds.items():
        print("   ",x,t)

roadtypes = {}
for u in railtypes[0]:
        ru = railunit[u]
        roadtype = ru.find("type").text
        if roadtype in roadtypes:
            roadtypes[roadtype].append(u)
        else:
            roadtypes[roadtype] = [u]

print("--------road types-------")
for rt,uuid in roadtypes.items():
    print("road type", rt)
    ruchilds = {}
    for u in uuid:
        print(uuids[u])
        ru = railunit[u]
        for child in ru:
            if child.tag in ruchilds:
                if child.text.isspace():
                    ruchilds[child.tag].append([c.text for c in child])
                else:
                    ruchilds[child.tag].append(child.text)
            else:
                if child.text.isspace():
                    ruchilds[child.tag] = [[c.text for c in child]]
                else:
                    ruchilds[child.tag] = [child.text]
    for x,t in ruchilds.items():
        print("   ",x,t)
                
print("--------transition types-------")
trans = {}
for rg in root.iter("RailGroup"):
    for rgi in rg.find("itemTransition"):
        uuid = int(rgi.find("uuid").text)
        trans[uuid] = railunit[uuid]

for uuid,ru in trans.items():
    print(uuid,uuids[uuid])
            
#slots = {}
#print("RailDictionary")
#for rd in root.iter("railDictionary"):
#    for i,u in enumerate(rd.findall("uuid/unsignedLong")):
#        uuid = int(u.text)
#        slots[uuid] = i
#        if uuid in uuids:
#            print(uuid,uuids[uuid])
#        elif uuid in uuids["Unknown"]:
#            print(uuid,"Unknown")
#        else:
#            print(uuid,"SuperUnknown")
#print(names["Unknown"])
#print(names["R05_CLIFF_START"])
    
# for rg in root.iter("RailGroup"):
#     print (rg[0].text)
#     for el_uuid in rg.iter("uuid"):
#         uuid = int(el_uuid.text)
#         if uuid in uuids:
#             print("   rail unit:", uuids[uuid])
#         else:
#             print("   unknown rail unit:", uuid)

            
#for uuid,ru in railunit.items():
#    print (uuid,uuids[uuid],slots[uuid])
#    print (ru.find("decoratedRailType").text, RailType(int(ru.find("decoratedRailType").text)))
#    print (ru.find("flag").text, TrackType(int(ru.find("flag").text)))
            
