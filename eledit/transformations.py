import math
from random import randint
        
def roll(chance = 0.5):
    roll = randint(0, 999)
    return roll < (1000 * chance)

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

def smoothen(heightsList, tolerance = 0.04, minLen = 5, maxLen = 60, flatten = True):
    def findSlopes(heightsList, tol = 0.04, minL = 5, maxL = 60, fltn = False):
        '''Extracts slopes from heights list.'''
        def calcSlope(s):
            '''Returns the slope between the first and last item in a list of height data.'''
            x1, x2, z1, z2 = getCoords(s)      
            slope = (z2-z1)/(x2-x1)
            return slope

        def delta_x(heights):
            if len(heights) < 2:
                delta = 0
            else:
                delta = heights[-1][0] - heights[0][0]
            return delta

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
            
        heights = heightsList.copy()                                                    #Making a shallow copy of the heights list, so that we retain an unmodified original in case something horrible would occur.
        slopes = []                                                                     #This list will contain all the slopes.
        
        while len(heights) >= minL:                                                     #A slope needs at least minLen height data points
            currentSlope = heights[:minL]
            heights = heights[minL:]
            slope = calcSlope(currentSlope)                                             #calculate the slope of the section you just grabbed

            while len(heights) >= minL and len(currentSlope) <= maxL :                  #While there are at least minLen more items in the heights list and currentSlope is not greater maxLen...
                nextItem = heights[0]                                                   #grab the next point of the heights list...
                if abs(calcSlope([currentSlope[-1]]+[nextItem])-slope) < tol:           #...and if the slope difference between that point and the previous point is within the tolerance value... 
                    currentSlope.append(heights.pop(0))#...we add it to the slope.
                else:
                    break                                                               #if not, we're done with this slope and break the loop.

            if len(heights) < minL:                                                     #If less than minLen height data points remain though...
                currentSlope += heights                                                 #...we add them to the slope...
                heights = []                                                            #...and clear the heights list.
            slopes.append(currentSlope)                                                 #...before adding the slope to the list of slopes.

        if fltn:
            slopes = flattenSlopes(slopes)                                              #Finally we flatten the slopes (optional)

        return slopes

    def roundCorners(segs):
        '''This function takes a list of segments and rounds off the corners using splines.'''
        curves = []
        
        for i, segA in list(enumerate(segs))[:-1]:
            #get the next segment
            segB = segs[i+1]

            #get start and endpoints
            dataA = [segA[0], segA[-1]]                           
            dataB = [segB[0], segB[-1]]

            #extract delta_x and slope
            for data in (dataA, dataB):
                delta_x = data[1][0] - data[0][0]
                delta_y = data[1][1] - data[0][1]
                slope = delta_y / delta_x
                data.append(slope)

            # calculate index
            indexA = -(len(segA)//2)
            indexB = (len(segB)//2)

            A = segA[indexA]
            B = segB[indexB]              

            #get (x, z, slope)
            A = (A[0], A[1], dataA[2])
            B = (B[0], B[1], dataB[2])

            curve = segA[indexA+1:]+segB[:indexB]

            #perform interpolation and add to [curves]
            splined = [spline(A, B, P) for P in curve]
            curves.append(splined)

        ungrouped_curves = [(item[0], item[1]) for curve in curves for item in curve]
        ungrouped_segs = [(item[0], item[1]) for seg in segs for item in seg]
        curve_x = [item[0] for item in ungrouped_curves]
        assembly = [slope for slope in ungrouped_segs if slope[0] not in curve_x]
        assembly += ungrouped_curves
        assembly.sort(key=lambda x: x[0])
        
        return assembly

    segs = findSlopes(heightsList, tol = tolerance, minL = minLen, maxL = maxLen, fltn = flatten )
    rounded = roundCorners(segs)
    output = rounded #[point for seg in segs for point in seg]
    return output#rounded    
        
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
    
