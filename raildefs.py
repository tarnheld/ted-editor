import xml.etree.ElementTree as ET
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

def getRailDict(path):
    tree = ET.parse(path)
    root = tree.getroot()
    uuids = {}
    names = { "Unknown" : [] }
    railtypes = {}
    railunit  = {}
    for ru in root.iter("RailUnit"):
        uuid = int(ru.find("uuid").text)
        rt   = int(ru.find("decoratedRailType").text)
             
        railunit[uuid] = ru
        uuids[uuid]    = "Unknown " + RailType(rt).name
        
        names["Unknown"].append(uuid) # unknown for now...
        
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


    for type,tuuids in railtypes.items():
        uidx = 1
        for uuid in tuuids:
            if uuid in names["Unknown"]:
                uuids[uuid] = uuids[uuid] + " " + str(uidx)
                uidx += 1
    return railtypes, uuids, railunit
