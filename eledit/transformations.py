import math
from random import randint

def gen_heights(r = 100):
    '''For testing purposes.'''
    my_list = [0 for i in range(0, r)]
    return my_list

def gen_structure(r = 100):
    structure = {'mod':[], 'height_data':[]}
    for i in range(0, 100):
        x = i
        y = 0
        slope = 0
        delta_slope = 0
        structure['mod'].append(y)
        structure['height_data'].append((x, y, slope, delta_slope))
    return structure
        

def roll(chance = 0.5):
    roll = randint(1, 100)
    if roll < 100 * chance:
        return True
    else:
        return False

def lim(structure, limits = (0.0, 1.0)):
    '''Calculates the limits indexes based on len of height_data'''
    def cap(item):
        if item < 0.0:
            item = 0.0
        elif item > 1.0:
            item = 1.0
        return item
    x_max = structure['height_data'][-1][0]
    start = x_max*cap(limits[0])
    stop = x_max*cap(limits[1]) 
    return start, stop

'''def noise(structure, magnitude = 0.1, freq = 5, chance = 0.5, limits = (0.1, 0.9)):
    x_pos, end = lim(structure, limits)
    
    while x_pos < end:
        if roll(chance):
            elev = randint(-10, 10)/(10/magnitude)
            feather = randint(2, 20)*abs(elev)
            structure = elevate(structure, x_pos, elev, 2, feather)
        x_pos += freq
    return structure

def roughen(structure):
    index = 0
    potenza = 4
    scale = 0.005
    v = variaion = 100
    for item in structure['height_data']:
        elev = ((randint(0, v)/v)**potenza)*scale
        neg = randint(0, 1)
        if neg:
            elev*=-neg         
        structure['mod'][index] += elev
        index += 1
    return structure'''

def wave(structure, elev = 1, freq = 30, chance = 0.5, limits = (0.1, 0.9)):
    x_pos, end = lim(structure, limits)

    while x_pos < end:
        if roll(chance):
            structure = elevate(structure, x_pos, elev, 0, freq/2)
        x_pos += freq
    return structure

def elevate(structure, x_pos, elev, brush = 5, feather = 10):
    '''Takes a list of height data and changes its elevation.'''
    index = 0
    for item in structure['height_data']:
        if x_pos - brush < item[0] < x_pos + brush: #if brush
            structure['mod'][index] += elev
        elif x_pos - brush - feather < item[0] < x_pos + brush + feather: #if feather
            if item[0] > x_pos:
                delta_x = item[0] - (x_pos + brush)
            else:
                delta_x = x_pos - (item[0] + brush) 
            y_factor = adj(delta_x, feather)
            structure['mod'][index] += elev*y_factor
        index += 1
    return structure

def smooth(structure, start, stop, factor = 0.5):
    def interpolate(start, stop, x):
        x1, y1 = start[0], start[1]
        x2, y2 = stop[0], stop[1]
        y = y1+(y2 - y1) * (x - x1) / (x2 - x1)
        return y
        
    index = 0
    indexes = []
    for item in structure['height_data']:
        if start < item[0] < stop:
            indexes.append(index)
        index += 1
    print(indexes)

    x_start = structure['height_data'][indexes[0]][0:2]
    x_stop = structure['height_data'][indexes[-1]][0:2]

    for index in range(indexes[0], indexes[-1]+1):
        x = structure['height_data'][index][0]
        y = structure['height_data'][index][1]
        elev = interpolate(x_start, x_stop, x) - y
        structure['mod'][index] += elev * factor

    return structure

def feather(feather, f_mode, delta_x):

    def mode_0(feather, delta_x):
        cos = math.cos(math.radians(30))
        flip = 0
        if delta_x > feather/2:
            delta_x = feather-delta_x
            flip = 1
        opp = 2 * cos * delta_x / feather
        f = adj = abs(flip-((1-(opp**2))**0.5))

        return f

    def mode_1(feather, delta_x):
        f = 1 - delta_x / feather
        return f

    def mode_2(feather, delta_x):
        f = abs(1 - (delta_x / feather)**2)**0.5
        return f

    def mode_3(feather, delta_x):
        f = 1-abs(1 - (1-(delta_x / feather))**2)**0.5
        return f

    def sine(feather, delta_x):
        pi = math.pi
        amp = 0.5
        freq = 0.5
        phase = pi/2
        x = delta_x/feather
        f = 0.5 + amp*math.sin(2*pi*freq*x+phase)
        return f
        

    modes = [sine, mode_1, mode_2, mode_3]
    f = modes[f_mode](feather, delta_x)
    return f

def flatten(A, B, P):
    #interpolation
    P[1] = (P[0]-A[0])/(B[0]-A[0])*(B[1]-A[1])+A[1]
    return (P[0], P[1], P[0], P[1])

def spline(A, B, P): #A, B = (x, z, slope), P = (x0, z0, x1, z1)
    #cubic hermite interpolation
    # t = ((pA, mA), (pB, mB))
    Ax, Az, m0 = A[0], A[1], A[2]
    Bx, Bz, m1 = B[0], B[1], B[2]
    Px = P[0]

    #convert data to tA = tB = 1
    p0 = 0
    p1 = (Bz-Az)/(Bx-Ax) #scale vertical data
    t = (Px-Ax)/(Bx-Ax) #scale horizontal P location

    #interpolation p(t)
    i = (t**3-2*t**2+t)*m0
    ii = (-2*t**3+3*t**2)*p1
    iii = (t**3-t**2)*m1
    p = i + ii + iii

    Pz = p*(Bx-Ax)+Az
    
    return (Px, Pz, Px, Pz)

def smoothen(heightsList):
    def findSlopes(heightsList, tolerance = 0.04, minLen = 6, maxLen = 18, flatten = True):
        '''Extracts slopes from heights list.'''
        def calcSlope(s):
            '''Returns the slope between the first and last item in a list of height data.'''
            x1, x2, z1, z2 = getCoords(s)      
            slope = (z2-z1)/(x2-x1)
            return slope

        def flattenSlope(s):
            '''Flattens a list of height data between the first and last item.'''
            flatSlope = []
            x1, x2, z1, z2 = getCoords(s)
            slope = calcSlope(s)
            for item in s:
                z = z1+(item[0]-x1)*slope
                flatSlope.append((item[0], z))
            return flatSlope

        def flattenSlopes(slopes):
            '''Flattens the slopes in a list of slopes.'''
            i = 1
            for slope in slopes[1:-1]: #Exclude first and last slope
                prevSlope = slopes[i-1]
                slope = flattenSlope([prevSlope[-1]]+slope)[1:]         
                slopes[i] = slope
                i += 1
            return slopes
                       
        def getCoords(s):
            '''Returns x and z values of the first and the last item in a list of height data.'''
            x1 = s[0][0]
            z1 = s[0][1]
            x2 = s[-1][0]
            z2 = s[-1][1]
            return x1, x2, z1, z2
            
        heights = heightsList.copy()        #Making a shallow copy of the heights list, so that we retain an unmodified original in case something horrible would occur.
        slopes = []                         #This list will contain all the slopes.
        while len(heights) >= minLen:             #A slope needs at least minLen height data points
            currentSlope = heights[:minLen]      #grab the first minLen height data points from the heights list...
            heights = heights[minLen:]           #...and remove them from the heights list
            slope = calcSlope(currentSlope) #then calculate the slope of the section you just grabbed
            
            while len(heights) >= minLen and len(currentSlope) <= maxLen :        #While there are at least minLen more items in the heights list and currentSlope is not greater maxLen...
                nextItem = heights[0]       #grab the next point of the heights list...
                if abs(calcSlope([currentSlope[-1]]+[nextItem])-slope) < tolerance:     #...and if the slope difference between that point and the previous point is within the tolerance value... 
                    currentSlope.append(heights.pop(0))                                 #...we add it to the slope.
                else:
                    break                   #if not, we're done with this slope and break the loop.
            if len(heights) < minLen:            #If less than minLen height data points remain though...
                currentSlope += heights     #...we add them to the slope...
                heights = []                #...and clear the heights list.
            slopes.append(currentSlope)     #...before adding the slope to the list of slopes.

        if flatten:
            slopes = flattenSlopes(slopes)      #Finally we flatten the slopes (optional)
            
        return slopes

    def roundCorners(slopes):
        '''This function takes a list of slopes and rounds off the corners.'''
        def generateRanges(slopes):
            '''This generates a list of ranges used when constructing the guides.'''
            def guideRange(slope):
                g1 = (len(slope)-1)//2
                g2 = (len(slope)-1-g1)
                return g1, g2
            
            ranges = []
            for slope in slopes:
                g1, g2 = guideRange(slope)
                if len(ranges) == 0:
                    ranges.append(g1)
                else:
                    if g1 < ranges[-1]:
                        ranges[-1] = g1           
                if len(ranges) != len(slopes):
                    ranges.append(g2)
                elif g2 < ranges[0]:
                    ranges[0] = g2
            return ranges

        def generateWeights(curve, gRange):
            '''This generates a weight for each point in a curve.'''
            x0 = curve[0][0]
            xCenter = curve[gRange][0]
            x1 = curve[-1][0]
            for item in curve:
                x = item[0]
                if x < xCenter:
                    xFrac = (x-x0)/(xCenter-x0)
                else:
                    xFrac = (x1-x)/(x1-xCenter)
                weight = 0.63*xFrac
                item.append(weight)
            return curve

        ranges = generateRanges(slopes)

        i = 1
        curves = []
        for slope in slopes[1:]:
            gRange = ranges[i]
            prevSlope = slopes[i-1]
            curve = prevSlope[-gRange-1:]+slope[:gRange]
            curve = [[x, z] for (x, z) in curve]
            i += 1
            
            curve = generateWeights(curve, gRange)

            guides = [] #(x0, z0, x1, z1, slope)
            counter = 0
            while counter < gRange:
                x0 = curve[counter][0]
                z0 = curve[counter][1]
                x1 = curve[counter-gRange][0]
                z1 = curve[counter-gRange][1]
                slope = (z1-z0)/(x1-x0)
                guides.append((x0, z0, x1, z1, slope))
                counter += 1

            #printGuides(guides)

            for item in curve:
                x = item[0]
                z = item[1]
                w = item[2]
                zGuide = None
                for g in guides:
                    x0, z0, x1, z1, slope = g[0], g[1], g[2], g[3], g[4]
                    if x0 <= x <= x1:
                        zX = z0 + (x-x0)*slope
                        if zGuide == None or abs(zX - z) > abs(zGuide-z) and x0 <= x <=x1:
                            zGuide = zX
                    elif zGuide == None:
                        zGuide = z
                z += w*(zGuide-z)
                item[1] = z
                            
            curves.append(curve)

        ungrouped_curves = [(item[0], item[1]) for curve in curves for item in curve]
        ungrouped_slopes = [(item[0], item[1]) for slope in slopes for item in slope]
        curve_x = [item[0] for item in ungrouped_curves]
        assembly = [slope for slope in ungrouped_slopes if slope[0] not in curve_x]
        assembly += ungrouped_curves
        assembly.sort(key=lambda x: x[0])
        
        return assembly

    slopes = findSlopes(heightsList)
    rounded = roundCorners(slopes)
    return rounded    
        
    
        
    
