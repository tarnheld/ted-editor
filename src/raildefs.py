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

def getRailRoot(path):
    tree = ET.parse(path)
    root = tree.getroot()
    return root
def getRailDict(root):
    names = {}
    railtypes = {}
    railunit  = {}
    for ru in root.iter("RailUnit"):
        uuid = int(ru.find("uuid").text)
        railunit[uuid] = ru

        rt   = int(ru.find("decoratedRailType").text)

        if rt in railtypes:
            railtypes[rt].append(uuid)
        else:
            railtypes[rt] = [uuid]

        names[uuid] = RailType(rt).name + "_" + str(len(railtypes[rt]))

    # get names from railgroup items
    for rgi in root.iter("RailGroupItem"):
        uuid = int(rgi.find("uuid").text)
        name =     rgi.find("name").text
        names[uuid] = name
        
    return railtypes, names, railunit

def getTransitionTypes(root):
    trans = {}
    for rg in root.iter("RailGroup"):
        for rgi in rg.find("itemTransition"):
            uuid = int(rgi.find("uuid").text)
            trans[uuid] = (uuid,rg)
    return trans

def getUnitLength(railunit):
    return float(railunit.find("unitLength").text)
def getUnitWidths(railunit):
    widths = []
    for mw in railunit.findall("modelWidth/float"):
            widths.append(float(mw.text))
    return widths

def getRailUnitIndex(root):
    index=[]
    inverse={}
    for rd in root.iter("railDictionary"):
        for idx,uuid in enumerate(rd.find("uuid")):
            uuid = int(uuid.find("unsignedLong").text)
            index.append(uuid)
            inverse[uuid] = idx
    return index,inverse
