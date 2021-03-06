import colorsys
import math as m
import pickle
import sys
import time
import collections
import tkinter as tk
from copy import deepcopy
from enum import Enum
from tkinter import filedialog

import numpy as np
from PIL import ImageTk, Image

import src.linalg as la
import src.read_ted as ted
from src import CanvasX as CX, raildefs

import src.biarc as biarc

from src.fsm import SavedFSM

######################################################################
class Heightmap:
    def __init__(self, file, half_width):
        self.heightmap = np.load(file)
        self.half_width = half_width
        print("loaded heightmap from", file)
    def heightAt(self,p):
        def interpol_bi(im, x, y):
            x = np.asarray(x)
            y = np.asarray(y)
            
            x0 = np.floor(x).astype(int)
            x1 = x0 + 1
            y0 = np.floor(y).astype(int)
            y1 = y0 + 1
            
            x0 = np.clip(x0, 0, im.shape[1] - 1)
            x1 = np.clip(x1, 0, im.shape[1] - 1)
            y0 = np.clip(y0, 0, im.shape[0] - 1)
            y1 = np.clip(y1, 0, im.shape[0] - 1)
            
            Ia = im[y0, x0]
            Ib = im[y1, x0]
            Ic = im[y0, x1]
            Id = im[y1, x1]
            
            wa = (x1 - x) * (y1 - y)
            wb = (x1 - x) * (y - y0)
            wc = (x - x0) * (y1 - y)
            wd = (x - x0) * (y - y0)
        
            return wa * Ia + wb * Ib + wc * Ic + wd * Id
        # 2d numpy array containing the data
        hmm = self.heightmap["heightmap"]
        # height values corresponding to 0 and 1 in the height map
        hme = self.heightmap["extents"]   
        hmw = hmm.shape[0]
        hmh = hmm.shape[1]
        ex = self.half_width
        ip = (p + ex) / (2 * ex) * la.coords(hmw, hmh)
        hs = interpol_bi(hmm, ip[0], ip[1])
        return hs * hme[1] + (1 - hs) * hme[0]
######################################################################
# global variables
the_scenery = {
    "Andalusia"   : {"id": 3,
                     "ct": "hm/andalusia-contours.npz",
                     "npz": "hm/andalusia.npz",
                     "ex": 3499.99975586,
                     "rd": "rd/andalusia.raildef"},
    "Eifel"       : {"id": 2,
                     "ct": "hm/eiffel-contours.npz",
                     "npz": "hm/eiffel.npz",
                     "ex": 9600.0,
                     "rd": "rd/eifel.raildef"},
    "Eifel Flat"  : {"id": 5,
                     "ct": "hm/eiffel-flat-contours.npz",
                     "npz": "hm/eiffel-flat.npz",
                     "ex": 9600.0,
                     "rd": "rd/eifel-flat.raildef"},
    "Death Valley": {"id": 1,
                     "ct": "hm/death-valley-contours.npz",
                     "npz": "hm/death-valley.npz",
                     "ex": 3592.67358398,
                     "rd": "rd/death-valley.raildef"},
}
the_scenery = collections.OrderedDict(sorted(the_scenery.items()))
the_heightmap = Heightmap(the_scenery["Andalusia"]["npz"],
                          the_scenery["Andalusia"]["ex"])

###########################################################################
# enable imports in third party directory eledit
sys.path.append("eledit")
import eledit.eledit as eed
import eledit.readfile_2 as eed_rf2
import src.upload_ted as ut

###########################################################################
class Biarc:
    def __init__(self, p0, t0, p1, t1, r):
        self.p0 = p0
        self.p1 = p1
        self.t0 = la.unit(t0)
        self.t1 = la.unit(t1)
        self.r = r
        self.JC = biarc.biarc_joint_circle(p0, t0, p1, t1)
        self.J = biarc.biarc_joint_point(self.JC, r)
        self.Jt = biarc.biarc_tangent_at_joint_point(self.J, p0, t0)
        self.h1 = biarc.bezier_circle_from_two_points_and_tangent(self.p0, self.J, self.t0)
        self.h2 = biarc.bezier_circle_from_two_points_and_tangent(self.J, self.p1, self.Jt)
        self.cp1 = biarc.circleparam(self.p0, self.J, self.t0)
        self.cp2 = biarc.circleparam(self.J, self.p1, self.Jt)
    def joint(self):
        return self.J
    def circleparameters(self):
        c1, k1, a1, l1 = self.cp1
        c2, k2, a2, l2 = self.cp2

        return c1, k1, a1, l1, c2, k2, a2, l2
    def length(self):
        c1, k1, a1, l1 = self.cp1
        c2, k2, a2, l2 = self.cp2
        return l1 + l2
    def offset(self, o):
        b = deepcopy(self)
        b.h1 = biarc.offset_circle(b.h1, o)
        b.h2 = biarc.offset_circle(b.h2, o)
        b.JC = biarc.offset_circle(b.JC, o)

        b.p0 = biarc.bezier2_h(0, b.h1)
        b.J = biarc.bezier2_h(1, b.h1)
        b.p1 = biarc.bezier2_h(1, b.h2)

        b.cp1 = biarc.circleparam(b.p0, b.J, b.t0)
        b.cp2 = biarc.circleparam(b.J, b.p1, b.Jt)
        return b
    def transform(self, xform):
        self.p0 = biarc.transform_point(self.p0, xform)
        self.p1 = biarc.transform_point(self.p1, xform)
        self.t0 = la.unit(biarc.transform_vector(self.t0, xform))
        self.t1 = la.unit(biarc.transform_vector(self.t1, xform))
        self.JC = biarc.transform_bezier_circle(self.JC, xform)
        self.J = biarc.transform_point(self.J, xform)
        self.Jt = la.unit(biarc.transform_vector(self.Jt, xform))
        self.h1 = biarc.transform_bezier_circle(self.h1, xform)
        self.h2 = biarc.transform_bezier_circle(self.h2, xform)
        self.cp1 = biarc.circleparam(self.p0, self.J, self.t0)
        self.cp2 = biarc.circleparam(self.J, self.p1, self.Jt)
    def t_and_hs_of_rsl(self, rsl):
        c1, k1, a1, l1, c2, k2, a2, l2 = self.circleparameters()
        s = rsl * (l1 + l2)
        rs = 0
        if s < l1:
            rs = s / l1
            t = biarc.bc_arclen_t(rs, a1)
            return t, self.h1
        else:
            rs = (s - l1) / l2
            t = biarc.bc_arclen_t(rs, a2)
            return t, self.h2
    def eval(self, rsl):
        """eval biarc at rsl from 0 to 1, meaning arc length/total biarc length"""
        t, hs = self.t_and_hs_of_rsl(rsl)
        return biarc.bezier2_h(t, hs)
    def eval_t(self, rsl):
        """eval biarc tangent at rsl from 0 to 1, meaning arc length/total biarc length"""
        t, hs = self.t_and_hs_of_rsl(rsl)
        return la.unit(biarc.bezier2_dr(t, hs))
    def evalJ(self, t):
        """eval joint circle from -1 to 1"""
        return biarc.circle2_h(t, self.JC)
    def dist_sqr_at_l(self, p):
        """return squared distance and arc length at nearest point to p on biarc"""
        p1, t1, cha1 = biarc.point_on_circle(self.p0, self.J, self.t0, p)
        p2, t2, cha2 = biarc.point_on_circle(self.J, self.p1, self.Jt, p)
        c1, k1, a1, l1, c2, k2, a2, l2 = self.circleparameters()
        mind, minl = None, None
        if 0 < t1 < 1:
            mind = la.norm2(p1 - p)
            aa1 = 2 * m.acos(cha1)
            minl = aa1 / abs(k1) if k1 != 0 else t1 * l1
        if 0 < t2 < 1 and (not mind or la.norm2(p2 - p) < mind):
            mind = la.norm2(p2 - p)
            aa2 = 2 * m.acos(cha2)
            minl = l1 + (aa2 / abs(k2) if k2 != 0 else t2 * l2)
        return mind, minl
###########################################################################
class Straight:
    def __init__(self, p0, p1):
        self.p0 = p0
        self.p1 = p1
    def length(self):
        return la.norm(self.p1 - self.p0)
    def offset(self, o):
        pt = la.perp2ccw(la.unit(self.p1 - self.p0))
        return Straight(self.p0 + pt * o, self.p1 + pt * o)
    def eval(self, t):
        return biarc.lerp(t, self.p0, self.p1)
    def eval_t(self, t):
        return la.unit(self.p1 - self.p0)
    def dist_sqr_at_l(self, p):
        """return squared distance to p and length of segment at nearest point"""
        def line_seg_distance_sqr_and_length(p0, p1, p):
            chord = p0 - p1
            top = p0 - p
            pa = la.para(top, chord)
            if la.dot(pa, chord) < 0:
                return la.norm2(p - p0), 0  # distance to first point
            elif la.norm2(chord) < la.norm2(pa):
                return la.norm2(p - p1), la.norm(chord)  # distance to last point
            else:
                return la.norm2(la.perp(top, chord)), la.norm(pa)  # distance to line segment
        return line_seg_distance_sqr_and_length(self.p0, self.p1, p)
    def transform(self, xform):
        self.p0 = biarc.transform_point(self.p0, xform)
        self.p1 = biarc.transform_point(self.p1, xform)
    def circleparameters(self):
        c1, k1, a1, l1 = (0, 0), 0, 0, la.norm(self.p0 - self.p1)
        c2, k2, a2, l2 = (0, 0), 0, 0, 0

        return c1, k1, a1, l1, c2, k2, a2, l2
###########################################################################
class CCPoint:
    def __init__(self, point=la.coords(0, 0), tangent=None):
        self.point = point
        self.tangent = tangent
###########################################################################
class SegType(Enum):
    Straight = 1
    Biarc = 2
###########################################################################
def canvas_create_circle(canvas, p, r, **kw):
    "convenience canvas draw routine"
    return canvas.create_oval([p[0] - r, p[1] - r, p[0] + r, p[1] + r], **kw)
###########################################################################
class Banking:
    def __init__(self, angle=0, prev_len=0, next_len=0):
        self.angle = angle
        self.prev_len = prev_len
        self.next_len = next_len
    def reverse(self):
        self.angle = -self.angle
        self.prev_len, self.next_len = self.next_len, self.prev_len
###########################################################################
class CCSegment:
    def __init__(self, p1, p2, type=SegType.Straight, biarc_r=0.5):
        self.ps = p1
        self.pe = p2
        self.type = type
        self.biarc_r = biarc_r
        self.banking = [Banking(), Banking()]
        self.heights = [[], []]
        self.seg = None
        self.heights_need_update = True

        self.setup()
    def setup(self):
        if self.type is SegType.Straight:
            self.ps.tangent = self.pe.tangent = la.unit(self.pe.point - self.ps.point)
            self.seg = Straight(self.ps.point, self.pe.point)
            #self.np = 2
        else:
            if self.pe.tangent is None:
                self.pe.tangent = la.unit(la.refl(self.ps.tangent,
                                                  self.ps.point - self.pe.point))

            self.seg = Biarc(self.ps.point, self.ps.tangent,
                             self.pe.point, self.pe.tangent,
                             self.biarc_r)
        self.heights_need_update = True
    def offset(self,o):
        return self.seg.offset(o)
    def length(self):
        return self.seg.length()
    def eval(self,t):
        return self.seg.eval(t)
    def eval_t(self,t):
        return self.seg.eval_t(t)
    def shapeparameters(self):
        """ returns end points, lengths, curvature, center of circle of subsegments"""
        divl = 4.0
        mindiv = 6
        if self.type is SegType.Straight:
            l = self.length()
            div = max(mindiv, int(m.ceil(l / divl)))
            return [(self.pe.point, l, div, 0, la.coords(0, 0))]
        else:
            c1, k1, a1, l1, c2, k2, a2, l2 = self.seg.circleparameters()
            div1 = max(mindiv, int(m.ceil(l1 / divl)))
            div2 = max(mindiv, int(m.ceil(l2 / divl)))
            return [(self.seg.J, l1, div1, k1, c1), (self.pe.point, l2, div2, k2, c2)]
    def num_subsegments(self):
        return len(self.shapeparameters())
    def set_heights(self, heights):
        self.heights = heights
        self.heights_need_update = False
        sys.stdout.flush()
    def calc_heights(self, heightmap):
        sps = self.shapeparameters()

        hs = heightmap.heightAt(self.ps.point)
        for j, sp in enumerate(sps):
            hgts = []
            ep, l, div, k, center = sp
            he = heightmap.heightAt(ep)
            for d in range(div):
                hgts.append(hs * (div - d) / div + he * d / div)
            self.heights[j] = hgts
            hs = he
        self.heights_need_update = False
    def reverse(self):
        # reverse endpoints from point array
        self.ps, self.pe = self.pe, self.ps
        # reverse biarc parameter
        self.biarc_r = 1 - self.biarc_r
        # banking and height data reverse
        self.banking.reverse()
        for b in self.banking:
            b.reverse()
        self.heights.reverse()
        for h in self.heights:
            h.reverse()
        self.setup()
        #self.heights_need_update = False
    def transform(self, xform):
        self.seg.transform(xform)
        # TODO: scale offset without altering width
        #for i, p in enumerate(self.poly):
        #    self.poly[i] = biarc.transform_point(p, xform)
###########################################################################
def offsetLine(segment, ts, te, o, n):
    so = segment.offset(o)
    ots = ts
    ote = te
    if isinstance(segment, Biarc):
        c1, k1, a1, l1, c2, k2, a2, l2 = segment.circleparameters()
        oc1, ok1, oa1, ol1, oc2, ok2, oa2, ol2 = so.circleparameters()
        ss = ts * (l1 + l2)
        se = te * (l1 + l2)
        if ss < l1:
            oss = ss / l1 * ol1
            ots = oss / (ol1 + ol2)
        else:
            oss = ol1 + (ss - l1) / l2 * ol2
            ots = oss / (ol1 + ol2)
        if se < l1:
            ose = se / l1 * ol1
            ote = ose / (ol1 + ol2)
        else:
            ose = ol1 + (se - l1) / l2 * ol2
            ote = ose / (ol1 + ol2)

    cas = []
    nn = max(n, 2)
    for j in range(nn):
        # TODO: distribute points evenly over both segments of biarc
        t = biarc.lerp(j / (nn - 1), ots, ote)
        p = so.eval(t)
        cas.append(p)
    return cas
def offsetPolygon(segment, ts, te, o1, o2, n):
    return offsetLine(segment, ts, te, o1, n) + offsetLine(segment, te, ts, o2, n)
###########################################################################
class ControlCurve:
    def __init__(self,width):
        self.width      = width
        self.scenery    = None
        self.isOpen     = True
        self.point      = []
        self.segment    = []
        self.checkpoint = []
        self.road       = []
        self.deco       = []
    def __segmentsof(self, cp):
        return [s for s in self.segment if s.ps is cp or s.pe is cp]
    def __neighborsof(self, seg):
        return [s for s in self.segment if seg.ps is s.pe or seg.pe is s.ps]
    def __setupCp(self, cp):
        so = self.__segmentsof(cp)
        for s in so:
            if s.type is SegType.Straight:
                return self.__setupSeg(s)
            else:
                s.setup()
        return so
    def __setupSeg(self, seg):
        seg.setup()
        nb = self.__neighborsof(seg)
        for s in nb:
            s.setup()
        nb.append(seg)
        return nb

    def length(self):
        "total length of control curve"
        tl = 0
        for s in self.segment:
            tl += s.length()
        return tl
    def segmentAndTAt(self, l):
        seg = None
        tl = l
        sl = 0
        while not seg:
            for s in self.segment:
                sl = s.length()
                if tl > sl:
                    tl -= sl
                else:
                    seg = s
                    break
        t = tl / sl
        return seg, t
    def pointAt(self, l):
        s, t = self.segmentAndTAt(l)
        # print("pointAt",l,s,t,s.seg.eval(t))
        return s.eval(t)
    def tangentAt(self, l):
        s, t = self.segmentAndTAt(l)
        return s.eval_t(t)
    def pointAndTangentAt(self, l):
        s, t = self.segmentAndTAt(l)
        return s.eval(t), s.eval_t(t)
    def nearest(self, p):
        mind = 100000000
        minl = None
        tl = 0
        for s in self.segment:
            d, l = s.seg.dist_sqr_at_l(p)
            if d is not None and mind > d:
                mind = d
                minl = tl + l
                # print("shorter!",d,l)
            tl += s.seg.length()
        return minl

        # internal functions
    def fixedSegmentPoints(self, seg):
        # return all points that have to move to leave seg fixed
        cps = set()
        cps.add(seg.ps)
        cps.add(seg.pe)
        for s in self.__neighborsof(seg):
            if s.type is SegType.Straight:
                cps.add(s.ps)
                cps.add(s.pe)
        return cps
    # manipulation methods:
    def setTangent(self, cp, t):
        cp.tangent = t
        return self.__setupCp(cp)

    def setPointAndTangent(self, cp, p, t):
        cp.point = p
        cp.tangent = t
        return self.__setupCp(cp)

    def transformPoints(self, cps, xform):
        segments = []
        for cp in cps:
            cp.point = biarc.transform_point(cp.point, xform)
            cp.tangent = la.unit(biarc.transform_vector(cp.tangent, xform))
            so = self.__segmentsof(cp)
            for s in so:
                if s.type is SegType.Straight:
                    so.extend(self.__neighborsof(s))
            segments.extend(so)
        transformable = set()
        reset = set()
        for s in segments:
            if s in reset:
                reset.remove(s)
                transformable.add(s)
            else:
                reset.add(s)
        for s in transformable:
            s.transform(xform)
        for s in reset:
            self.__setupSeg(s)
        return reset.union(transformable)

    def movePoint(self, cp, vec):
        cp.point = cp.point + vec
        return self.__setupCp(cp)

    def moveJoint(self, seg, p):
        J = seg.seg.joint()
        p0 = seg.ps.point
        t0 = seg.ps.tangent
        p1 = seg.pe.point
        t1 = seg.pe.tangent
        tj = biarc.biarc_joint_circle_tangent(p0, t0, p1, t1)
        pc, t, a = biarc.point_on_circle(p0, p1, tj, p)
        seg.biarc_r = t
        seg.setup()
        d = pc - J  # displacement of joint point

        # print (pc,seg.seg.joint(),pc2,pc2-pc)

        return [seg], d[0], d[1]
    def tangentUpdateable(self, cp):
        for s in self.__segmentsof(cp):
            # print("     ",s.type,s.ps.point,s.pe.point)
            if s.type is SegType.Straight:
                return False
        return True

    # all manipulating functions return the affected segments
    def toggleOpen(self, *args):
        if self.isOpen:
            close = CCSegment(self.point[-1], self.point[0], SegType.Biarc, *args)
            self.segment.append(close)
            self.isOpen = False
            return close
        else:
            rem = self.segment.pop()
            self.isOpen = True
            return rem
    # returns the new point, the new segments and all affected segments
    def changeType(self, seg):
        if seg is self.segment[0]:  # first segment has to stay straight
            return []
        if seg.type is SegType.Biarc:
            for s in self.__neighborsof(seg):
                if s.type == SegType.Straight:
                    return []
            seg.type = SegType.Straight
        else:
            seg.type = SegType.Biarc
        return self.__setupSeg(seg)
    def insertPoint(self, seg, p, *args):
        """insert a control point and return the point, the newly created segment and the affected segments of this operation"""
        cp = CCPoint(p)
        if not self.point:  # first point
            self.point.append(cp)

            return cp, None, None
        if not self.segment and len(self.point) == 1:  # first segment
            args = [SegType.Straight if x == SegType.Biarc else x for x in args]
            newseg = CCSegment(self.point[0], cp, *args)
            self.point.append(cp)
            self.segment.append(newseg)
            return cp, newseg, newseg

        si = None
        if seg is None:  # append at end
            si = len(self.segment)
            newseg = CCSegment(self.point[si], cp, *args)
        else:  # split an existing segment
            pe, seg.pe = seg.pe, cp
            seg.setup()
            newseg = CCSegment(cp, pe, *args)
            si = self.segment.index(seg)

        self.point.insert(si + 2, cp)
        self.segment.insert(si + 1, newseg)

        aff = self.__setupSeg(newseg)

        return cp, newseg, aff
    def appendPoint(self, p, *args):
        return self.insertPoint(None, p, *args)
    def removePoint(self, cp):
        if cp is self.point[0] or cp is self.point[1]:
            return []  # don't remove first point or segment

        segs = self.__segmentsof(cp)

        if len(segs) > 1:
            segs[1].ps = segs[0].ps

        self.point.remove(cp)
        self.segment.remove(segs[0])

        aff = []
        if len(segs) > 1:
            aff = self.__setupSeg(segs[1])

        # add removed segs[0] to clear canvas items in redraw
        return [segs[0]] + aff
    def reverse(self):
        # reverse lists and reset start to old point 1 and segment 0
        self.point.reverse()
        self.point.insert(0, self.point.pop())
        self.point.insert(0, self.point.pop())

        self.segment.reverse()
        self.segment.insert(0, self.segment.pop())

        # reverse tangents
        for p in self.point:
            p.tangent = -p.tangent
            # print("point",p.point,p.tangent)

        # reverse start and end points in segments
        for s in self.segment:
            s.reverse()
            # print("segment",s.type,s.ps.point,s.pe.point)

        # print(self.pointAt(0))

        return self.segment  # all are affected
    def offsetPolygonAt(self, ls, le, o1, o2 = None):
        """ constructs an offset polygon of width w along the track from ls to le"""
        if o2 is None:
            o2 =  o1/2
            o1 = -o1/2
        tls = min(ls, le)
        tle = max(ls, le)
        sl = 0
        front = []
        back  = []
        for s in self.segment:
            sl = s.length()
            if tls < sl:
                ts = tls / sl if tls > 0  else 0
                te = tle / sl if tle < sl else 1
                front += offsetLine(s.seg, ts, te, o1, 16)
                back  += offsetLine(s.seg, ts, te, o2, 16)
            tls -= sl
            tle -= sl
            if tls <= 0 and tle <= 0:
                break
        back.reverse()
        return front + back
    def drawSegment(self, seg, canvas, inclText=True, **config):
        # returns a list of the cids of the canvas items or
        # None if segment isn't in the curve
        def segmentOffsetPoly(seg,width):
            front = []
            back  = []
            shapes = seg.shapeparameters()
            sl = seg.length()
            dl = 0
            o = width/2
            for j, shape in enumerate(shapes):
                ep, l, div, k, center = shape
                n = max(2,min(32,int(abs(3 * 2*k*m.pi)*l)))
                s = dl/sl
                e = (dl+l)/sl
                front += offsetLine(seg,s,e,-o,n)
                back  += offsetLine(seg,s,e,+o,n)
                dl += l
            back.reverse()
            return front+back
        def segmentDraw(seg, canvas, cids=None, **kw):
            if not cids: cids = []
            poly = segmentOffsetPoly(seg,self.width)
            cids.append(canvas.create_polygon([(x[0], x[1]) for x in poly], **kw))
            return cids
        def segmentText(seg, canvas, cids=None, **kw):
            if not cids: cids = []
            shapes = seg.shapeparameters()
            sl = seg.length()
            dl = 0
            for j, shape in enumerate(shapes):
                ep, l, div, k, center = shape
                if abs(k) < 1 / 300:
                    stext = "\n{:.0f}".format(l)
                else:
                    stext = "{:.0f} R\n{:.0f}".format(1 / k, l)
                h = seg.eval(0.5 * (dl+l) / sl)
                cids.append(canvas.create_text([h[0], h[1]],
                                               text=stext,
                                               tags="segment"))
                dl += l
            return cids
        def segmentExtra(self, canvas, cids=None, **kw):
            if not cids: cids = []
            # if self.type is SegType.Straight:
            #     return cids
            # t1 = biarc.bezier2_dr(1, self.seg.h1)
            # t2 = biarc.bezier2_dr(0, self.seg.h2)
            # J = self.seg.joint()
            
            # cids.append(canvas.create_line([self.ps.point[0], self.ps.point[1], self.pe.point[0], self.pe.point[1]],
            #                                tags="segment"))
            # cids.append(canvas.create_line([J[0], J[1], J[0] + t1[0], J[1] + t1[1]], tags="segment"))
            
            # cas = []
            # nn = 32
            # cj, kj, aj, lj = self.seg.circleparameters()
            # for j in range(nn):
            #     t = biarc.lerp(j / (nn - 1), -1, 1)
            #     tt = biarc.bc_arclen_t(abs(t), aj if t > 0 else 2 * m.pi - aj)
            #     p = self.seg.evalJ(tt if t > 0 else -tt)
            #     cas.append(p)
            # cids.append(canvas.create_line([(x[0], x[1]) for x in cas], tags="segment"))
            return cids


        if seg in self.segment:
            cids = segmentDraw(seg,canvas,**config)
            if inclText:
                segmentText(seg,canvas, cids, **config)
                # segmentExtra(seg,canvas,cids,**config)
            return cids
        else:
            return None
    def draw(self, canvas, inclText=True, **config):
        # returns a map of segments to a list of cids of the canvas items
        seg2cids = {}
        for s in self.segment:
            seg2cids[s] = self.drawSegment(s, canvas, inclText, **config)
        return seg2cids
    def importTed(self, tedfile):
        hdr     = ted.ted_data_to_tuple("header", ted.header, tedfile, 0)
        cps     = ted.ted_data_to_tuple_list("cp", ted.cp, tedfile,
                                             hdr.cp_offset, hdr.cp_count)
        banks   = ted.ted_data_to_tuple_list("bank", ted.segment, tedfile,
                                             hdr.bank_offset, hdr.bank_count)
        heights = ted.ted_data_to_tuple_list("height", ted.height, tedfile,
                                             hdr.height_offset, hdr.height_count)
        checkps = ted.ted_data_to_tuple_list("checkpoints", ted.checkpoint, tedfile,
                                             hdr.checkpoint_offset, hdr.checkpoint_count)
        road    = ted.ted_data_to_tuple_list("road", ted.road, tedfile,
                                             hdr.road_offset, hdr.road_count)
        deco    = ted.ted_data_to_tuple_list("decoration", ted.decoration, tedfile,
                                             hdr.decoration_offset, hdr.decoration_count)

        self.width   = hdr.road_width
        self.scenery = hdr.scenery
        
        self.isOpen = True
        self.point   = []
        self.segment = []
        self.road    = []
        self.deco    = []
        self.checkpoint = []
        def set_banking(banking, bank):
            banking.angle    = bank.banking
            banking.prev_len = bank.transition_prev_vlen
            banking.next_len = bank.transition_next_vlen

        def get_segment_heights(heights, bank):
            sh = [heights[j].height for j in range(bank.total_divisions,
                                                   bank.total_divisions + bank.divisions)]
            return sh

        for i, cp in enumerate(cps):
            prev = cps[i - 1]
            lx = (cps[i - 1].x - cp.x)
            ly = (cps[i - 1].y - cp.y)
            l = m.sqrt(lx * lx + ly * ly)
            if ted.SegType(cp.segtype) == ted.SegType.Straight:
                # straight segment
                p, seg, _ = self.appendPoint(la.coords(cp.x, cp.y), SegType.Straight)
                if seg:
                    set_banking(seg.banking[0], banks[i - 1])
                    hs = get_segment_heights(heights, banks[i - 1])
                    seg.set_heights([hs, []])
            # second arc of biarc segment
            segarc2 = (ted.SegType(cp.segtype) == ted.SegType.Arc2CCW or
                       ted.SegType(cp.segtype) == ted.SegType.Arc2CW)
            prevsegarc1 = (ted.SegType(prev.segtype) == ted.SegType.Arc1CCW or
                           ted.SegType(cp.segtype) == ted.SegType.Arc1CW)
            segnearly = ted.SegType(cp.segtype) == ted.SegType.NearlyStraight
            prevsegnearly = ted.SegType(prev.segtype) == ted.SegType.NearlyStraight

            if segarc2 or segnearly and (prevsegarc1 or prevsegnearly):
                p1    = la.coords(cps[i - 0].x, cps[i - 0].y)
                joint = la.coords(cps[i - 1].x, cps[i - 1].y)
                p0    = la.coords(cps[i - 2].x, cps[i - 2].y)

                dx = (cp.x - cp.center_x)
                dy = (cp.y - cp.center_y)

                if ted.SegType(cp.segtype) == ted.SegType.Arc2CCW:
                    t = la.unit(la.coords(dy, -dx))
                if ted.SegType(cp.segtype) == ted.SegType.Arc2CW:
                    t = la.unit(la.coords(-dy, dx))
                if ted.SegType(cp.segtype) == ted.SegType.NearlyStraight:
                    t = la.unit(la.coords(cp.x - prev.x, cp.y - prev.y))

                biarc_r = la.norm(joint - p0) / (la.norm(joint - p0) + la.norm(joint - p1))

                # biarc_r = bezier_circle_parameter(p0,p1,self.cc.point[-1].tangent,joint)

                p, seg, _ = self.appendPoint(la.coords(cp.x, cp.y), SegType.Biarc, biarc_r)
                p.tangent = t

                set_banking(seg.banking[0], banks[i - 2])
                set_banking(seg.banking[1], banks[i - 1])
                hs1 = get_segment_heights(heights, banks[i - 2])
                hs2 = get_segment_heights(heights, banks[i - 1])
                seg.set_heights([hs1, hs2])

        # ted file contains first point as last point if track is closed, fix it:
        if hdr.is_loop:
            self.segment[-1].pe = self.segment[0].ps
            self.segment[-1].setup()
            self.point.pop()  # remove duplicated last point
            self.isOpen = False

        for s in self.segment:  # quick fix for setup update madness while importing
            s.heights_need_update = False

        def neighborhood(iterable):
            iterator = iter(iterable)
            prev = None
            current = next(iterator)  # throws StopIteration if empty.
            for next_item in iterator:
                yield (prev, current, next_item)
                prev,current = current,next_item
            yield (prev, current, None)

        for c in checkps:
            self.checkpoint.append(c.vpos3d)
        for r in road:
            self.road.append(r)
        for d in deco:
            self.deco.append(d)
    def exportTed(self):
        # convert ted
        TedCp = ted.ted_data_tuple("cp", ted.cp)
        TedSeg = ted.ted_data_tuple("segment", ted.segment)
        segs = []
        total_l = 0
        total_div = 0
        total_heights = []
        eps = 2 ** -16

        # control points always include first point, first segment is always straight
        cps = [TedCp(segtype=int(ted.SegType.Straight.value),
                     x=self.segment[0].ps.point[0],
                     y=self.segment[0].ps.point[1],
                     center_x=0,
                     center_y=0)]

        for i, seg in enumerate(self.segment):
            if seg.heights_need_update:
                seg.calc_heights(the_heightmap)
            for j, sp in enumerate(seg.shapeparameters()):
                ep, l, div, k, center = sp
                bk   = seg.banking[j]
                hgts = seg.heights[j]
                bdiv = len(hgts)

                if seg.type is SegType.Straight:
                    segtype = ted.SegType.Straight
                elif abs(k) < eps:
                    segtype = ted.SegType.NearlyStraight
                elif j == 0:
                    segtype = ted.SegType.Arc1CW if k > 0 else ted.SegType.Arc1CCW
                elif j == 1:
                    segtype = ted.SegType.Arc2CW if k > 0 else ted.SegType.Arc2CCW

                cps.append(TedCp(segtype=int(segtype.value),
                                 x=ep[0],
                                 y=ep[1],
                                 center_x=center[0],
                                 center_y=center[1]))
                segs.append(TedSeg(banking=bk.angle,
                                   transition_prev_vlen=bk.prev_len,
                                   transition_next_vlen=bk.next_len,
                                   divisions=bdiv,
                                   total_divisions=total_div,
                                   vstart=total_l,
                                   vlen=l))
                # print("seg",l,div,total_div,l,total_l)
                total_l   += l
                total_div += bdiv
                total_heights.extend(hgts)

        TedHgt = ted.ted_data_tuple("height", ted.height)
        tedheights = []
        for h in total_heights:
            tedheights.append(TedHgt(height=h))
        # always include first height value as last (TODO: check if valid for open tracks)
        tedheights.append(tedheights[0])

        eldiff = max(total_heights) - min(total_heights)

        TedCheck = ted.ted_data_tuple("checkpoint", ted.checkpoint)
        chps = []
        for cp in self.checkpoint:
            chps.append(TedCheck(vpos3d = cp))

        # convert road railunits
        TedRoad = ted.ted_data_tuple("road", ted.road)
        roads = []  # holds new road elements, can be more or less than elements in self.road
        for r in self.road:  # all road elements in the imported ted file
            roads.append(TedRoad(uuid=r.uuid,
                                 side=r.side,
                                 vstart3d=r.vstart3d,
                                 vend3d=r.vend3d))

        # convert decoration railunits
        TedDeco = ted.ted_data_tuple("deco", ted.decoration)
        decos = []
        for d in self.deco:
            decos.append(TedDeco(uuid=d.uuid,
                                 side=d.side,
                                 vstart3d=d.vstart3d,
                                 vend3d=d.vend3d,
                                 tracktype=d.tracktype))

        # build new tedfile
        TedHdr = ted.ted_data_tuple("hdr", ted.header)
        hdr = TedHdr(id=b"GT6TED\0\0",
                     version=104,
                     scenery=self.scenery,
                     road_width=self.width,
                     track_width_a=self.width,
                     track_width_b=self.width,
                     track_length=total_l,
                     datetime=int(time.time()),
                     is_loop=0 if self.isOpen else 1,
                     home_straight_length=700,
                     elevation_diff=eldiff,
                     num_corners=4, # TODO: calc
                     finish_line=0,
                     start_line=412.0769348144531, # TODO: from raildef
                     empty1_offset=0,
                     empty1_count=0,
                     empty2_offset=0,
                     empty2_count=0,
                     empty3_offset=0,
                     empty3_count=0)

        tedsz = (ted.ted_data_size(ted.header) +
                 ted.ted_data_size(ted.cp) * len(cps) +
                 ted.ted_data_size(ted.segment) * len(segs) +
                 ted.ted_data_size(ted.height) * len(tedheights) +
                 ted.ted_data_size(ted.checkpoint) * len(chps) +
                 ted.ted_data_size(ted.road) * len(roads) +
                 ted.ted_data_size(ted.decoration) * len(decos))

        tedfile = bytearray(b'\x00' * tedsz)

        hdr = hdr._replace(scenery=self.scenery,
                           track_length=total_l,
                           elevation_diff=eldiff)
        hdrsz = ted.ted_data_size(ted.header)

        # o = ted.tuple_to_ted_data(hdr,ted.header,self.tedfile,0)
        hdr = hdr._replace(cp_offset=hdrsz, cp_count=len(cps))
        o = ted.tuple_list_to_ted_data(cps, ted.cp, tedfile, hdr.cp_offset, hdr.cp_count)
        hdr = hdr._replace(bank_offset=o, bank_count=len(segs))
        o = ted.tuple_list_to_ted_data(segs, ted.segment, tedfile, hdr.bank_offset, hdr.bank_count)
        hdr = hdr._replace(height_offset=o, height_count=len(tedheights))
        o = ted.tuple_list_to_ted_data(tedheights, ted.height, tedfile, hdr.height_offset, hdr.height_count)
        hdr = hdr._replace(checkpoint_offset=o, checkpoint_count=len(chps))
        o = ted.tuple_list_to_ted_data(chps, ted.checkpoint, tedfile, hdr.checkpoint_offset, hdr.checkpoint_count)
        hdr = hdr._replace(road_offset=o, road_count=len(roads))
        o = ted.tuple_list_to_ted_data(roads, ted.road, tedfile, hdr.road_offset, hdr.road_count)
        hdr = hdr._replace(decoration_offset=o, decoration_count=len(decos))
        o = ted.tuple_list_to_ted_data(decos, ted.decoration, tedfile, hdr.decoration_offset, hdr.decoration_count)

        ted.tuple_to_ted_data(hdr, ted.header, tedfile, 0)
        return tedfile

###########################################################################
def rgb2hex(rgb_tuple):
    """ convert an (R, G, B) tuple to #RRGGBB """
    return '#%02x%02x%02x' % tuple(int(x * 255) for x in rgb_tuple)
def hex2rgb(hexstring):
    """ convert #RRGGBB to an (R, G, B) tuple """
    hexstring = hexstring.strip()
    if hexstring[0] == '#': hexstring = hexstring[1:]
    if len(hexstring) != 6:
        raise ValueError("input #%s is not in #RRGGBB format" % hexstring)
    r, g, b = hexstring[:2], hexstring[2:4], hexstring[4:]
    r, g, b = [int(n, 16) for n in (r, g, b)]
    return r / 255, g / 255, b / 255
def rgb_lighten_saturate(rgb, dl,ds):
    hls = colorsys.rgb_to_hls(*rgb)
    hls = (hls[0], hls[1] + dl, hls[2] + ds)
    return colorsys.hls_to_rgb(*hls)
def hex_lighten_saturate(hexstr, dl, ds):
    hls = colorsys.rgb_to_hls(*hex2rgb(hexstr))
    hls = (hls[0], hls[1] + dl, hls[2] + ds)
    return rgb2hex(colorsys.hls_to_rgb(*hls))
def style_active(style, lighten, saturate, thicken):
    if "fill" in style and style["fill"]:
        style["activefill"] = hex_lighten_saturate(style["fill"], lighten, saturate)
    if "outline" in style:
        style["activeoutline"] = hex_lighten_saturate(style["outline"], lighten, saturate)
    if "width" in style:
        style["activewidth"] = style["width"] + thicken
    return style
def style_modified(style, lighten, saturate, thicken):
    style = deepcopy(style)
    if "fill" in style and style["fill"]:
        style["fill"] = hex_lighten_saturate(style["fill"], lighten, saturate)
    if "activefill" in style and style["activefill"]:
        style["activefill"] = hex_lighten_saturate(style["activefill"], lighten, saturate)
    if "outline" in style:
        style["outline"] = hex_lighten_saturate(style["outline"], lighten, saturate)
    if "activeoutline" in style:
        style["activeoutline"] = hex_lighten_saturate(style["activeoutline"], lighten, saturate)
    if "width" in style:
        style["width"] = style["width"] + thicken
    if "activewidth" in style:
        style["activewidth"] = style["activewidth"] + thicken
    return style
def style_config_mod(cfg, lighten, saturate, thicken):
    style = {}
    if "fill" in cfg and cfg["fill"][4]:
        style["fill"] = hex_lighten_saturate(cfg["fill"][4], lighten, saturate)
    if "outline" in cfg and cfg["outline"][3] is not cfg["outline"][4]:
        style["outline"] = hex_lighten_saturate(cfg["outline"][4], lighten, saturate)
    if "width" in cfg and cfg["width"][3] is not cfg["width"][4]:
        style["width"] = float(cfg["width"][4]) + thicken
    return style

###########################################################################
# helper class for text popups
class PopupEntry(tk.Toplevel):
    def __init__(self, master, pos, desc, cb, title=None):
        super().__init__(master)
        self.cb = cb

        self.transient(master)
        if title:
            self.title(title)
        self.geometry("+%d+%d" % (master.winfo_rootx() + pos[0],
                                  master.winfo_rooty() + pos[1]))
        self.l = tk.Label(self, text=desc)
        self.l.pack()
        self.e = tk.Entry(self)
        self.e.pack()
        self.initial_focus = self.e
        self.b = tk.Button(self, text='Ok', command=self.cleanup)
        self.b.pack()

        self.e.bind("<Return>", self.cleanup)
        self.e.focus_set()
    def cleanup(self, ev=None):
        value = self.e.get()
        self.destroy()
        self.cb(value)
class PopupAbout(tk.Toplevel):
    def __init__(self, master, pos, desc, title=None):
        super().__init__(master)
        self.transient(master)
        if title:
            self.title(title)
        self.geometry("+%d+%d" % (master.winfo_rootx() + pos[0],
                                  master.winfo_rooty() + pos[1]))
        self.l = tk.Label(self, text=desc)
        self.l.pack()
        self.b = tk.Button(self, text='Ok', command=self.cleanup)
        self.b.pack()
    def cleanup(self, ev=None):
        self.destroy()

###########################################################################
# unfinished:
###########################################################################
class RailManip(SavedFSM):
    def __init__(self, cc, canvas):
        self.cc = cc  # TODO: what if new curve is exported???!!!
        self.canvas = canvas

        self.movetag = "rail"

        s = Enum("States", "Idle Insert Move ChangeType RRem")
        tt = {
            # (State, tag or None, event (None is any event)) -> (State, callback)
            (s.Idle, None, "<ButtonPress-1>")    : (s.Insert, self.onRailInsertStart),
            (s.Insert, None, "<B1-Motion>")      : (s.Insert, self.onRailInsertUpdate),
            (s.Insert, None, "<ButtonRelease-1>"): (s.Idle, self.onRailInsertEnd),

            (s.Idle, "rail", "<ButtonPress-1>")  : (s.Move, self.onRailMoveStart),
            (s.Move, "rail", "<B1-Motion>")      : (s.Move, self.onRailMoveUpdate),
            (s.Move, "rail", "<ButtonRelease-1>"): (s.Idle, self.onRailMoveEnd),

            (s.Idle, "rail", "<Double-Button-1>"): (s.RRem, self.onRailRemove),
            (s.RRem, None, "<ButtonRelease-1>")  : (s.Idle, None),

            (s.Idle, "rail", "<ButtonPress-3>")  : (s.Idle, self.onChangeType),

        }
        super().__init__(s.Idle, tt, self.canvas)

        self.movestyle = {
            "width"  : 0,
            "outline": "#222222",
            "fill"   : "#9370DB",
            "tags"   : "rail"
        }

        # style_active(self.segstyle,  -0.1, 0,   2)
        style_active(self.movestyle, -0.1, 0.1, 3)
        # style_active(self.rotstyle,  -0.1, 0.1, 3)

        self.selstyle = style_modified(self.movestyle, -0.2, 0.2, 2)

        self.r_cidmap = {}
        self.imap = {}

        class EvInfo:
            pass
        self.info = EvInfo
        self.info.item = None

        self.type = tk.StringVar()

        self.cr = None
        self.scene = None
        self.rr = None
        self.rt = None
        self.uuids = None
        self.ru = None
        self.rui = None
    def switchScenery(self,scene):
        if self.scene is not None and self.scene is not scene:
            old_rui,old_rui_inv = raildefs.getRailUnitIndex(self.rr)
            new_rr = raildefs.getRailRoot(scene["rd"])
            new_rui,new_rui_inv = raildefs.getRailUnitIndex(new_rr)
            for r in self.cc.road:
                r.uuid = new_rui[old_rui_inv[r.uuid]]
        self.scene = scene
        raildef_file = scene["rd"]
        self.rr = raildefs.getRailRoot(raildef_file)
        self.rt, self.uuids, self.ru = raildefs.getRailDict(self.rr)

    def addMoveHandle(self, railitem):
        #print("poly from",railitem.vstart3d,"to",railitem.vend3d)
        try:
            unit = self.ru[railitem.uuid]
        except KeyError:
            return []
        
        #print(railitem.uuid,unit, raildefs.getUnitLength(unit))

        widths = raildefs.getUnitWidths(unit)
        print(widths)
        if railitem.side== 0 or railitem.side == 2:
            print(railitem.side, "l",widths[0], "r",widths[1])
            opl = self.cc.offsetPolygonAt(railitem.vstart3d,railitem.vend3d,-self.cc.width/2,-widths[0])
            opr = self.cc.offsetPolygonAt(railitem.vstart3d,railitem.vend3d,+self.cc.width/2,+widths[1])
        else:
            print(railitem.side, "l",widths[1], "r",widths[0])
            opl = self.cc.offsetPolygonAt(railitem.vstart3d,railitem.vend3d,-self.cc.width/2,-widths[1])
            opr = self.cc.offsetPolygonAt(railitem.vstart3d,railitem.vend3d,+self.cc.width/2,+widths[0])
            
        cids = [self.canvas.create_polygon([(p[0],p[1]) for p in opl], **self.movestyle),
                self.canvas.create_polygon([(p[0],p[1]) for p in opr], **self.movestyle)]
        return cids
    def removeHandles(self):
        self.canvas.delete(self.movetag)
    def addHandles(self):
        self.removeHandles()
        self.cp_cidmap = {}
        print(len(self.cc.road))
        for r in self.cc.road:
            print(len(r))
            cids = self.addMoveHandle(r)
            for cid in cids:
                self.r_cidmap[cid] = r
            self.imap[r] = cids
        self.canvas.tag_lower("rail","segment")
        sys.stdout.flush()

        minslotlen = 50
        #tex = m.fmod(self.cc.length(),minslotlen)
        #numslots = int((self.cc.length() - tex)/minslotlen + 2)
        numslots = int(self.cc.length()/minslotlen + 2)
        ex = (self.cc.length() - numslots*minslotlen)/numslots
        slotlen = minslotlen + ex

        
        try:
            self.canvas.delete("slots")
        except tk.TclError:
            pass

        for i in range(numslots):
            s = i*slotlen
            op, ot = self.cc.pointAndTangentAt(s)
            on = la.perp2ccw(ot)
            p1 = op + 30*la.unit(on)
            p2 = op - 30*la.unit(on)
            self.canvas.create_polygon([(p1[0],p1[1]),(p2[0],p2[1])],outline="black",tags="slots")


    def findClosest(self, ev):
        cx, cy = self.canvas.canvasxy(ev.x, ev.y)
        item = self.canvas.find_closest(cx, cy)[0]
        railitem = self.r_cidmap[item]
        return cx, cy, item, railitem

    def onRailRemove(self, ev):
        # self.historySave()
        cx, cy, item, ri = self.findClosest(ev)
        # print("RailRemove")
        # self.cr.rails.remove(ri)
        self.addHandles()

    def onChangeType(self, ev):
        cx, cy, item, ri = self.findClosest(ev)
        self.info.ri = ri
        self.type.set(ri.type)
        # self.info.type, menu = self.cr.typePopup(canvas)
        menu = tk.Menu(self.canvas, tearoff=0, takefocus=0)

        menu.add_radiobutton(label="Normal", value="NORMAL", variable=self.type, command=self.onChangeTypeEnd)
        menu.add_radiobutton(label="Runoff Right", value="R_RUNOFF_RIGHT", variable=self.type,
                             command=self.onChangeTypeEnd)
        menu.add_radiobutton(label="Runoff Left", value="R_RUNOFF_LEFT", variable=self.type,
                             command=self.onChangeTypeEnd)

        menu.tk_popup(ev.x_root, ev.y_root, entry=0)

    def onChangeTypeEnd(self):
        # print("new type",self.type.get())
        self.info.ri.type = self.type.get()
        self.addHandles()

    def onRailMoveStart(self, ev):
        cx, cy, item, ri = self.findClosest(ev)
        self.info.ri = ri
        self.info.item = item

        pass
    def onRailMoveUpdate(self, ev):
        cx, cy = self.canvas.canvasxy(ev.x, ev.y)
        l = self.cc.nearest(la.coords(cx, cy))
        ol = self.info.ri.length()
        self.info.ri.setLength(l)
        op, ot = self.cc.pointAndTangentAt(ol)
        p, t = self.cc.pointAndTangentAt(l)

        a1 = m.atan2(ot[0], ot[1])
        a2 = m.atan2(t[0], t[1])

        angle = a1 - a2

        xform = la.identity()
        xform = la.mul(xform, la.translate(p[0], p[1]))
        xform = la.mul(xform, la.rotate(angle, 0, 0, 1))
        xform = la.mul(xform, la.translate(-op[0], -op[1]))

        cids = self.imap[self.info.ri]
        self.canvas.apply_xform(cids, xform)
        # self.canvas.move(self.info.item,p[0] - op[0],p[1] - op[1])
        pass
    def onRailMoveEnd(self, ev):
        pass
    def onRailInsertStart(self, ev):
        cx, cy = self.canvas.canvasxy(ev.x, ev.y)
        self.info.p = la.coords(cx, cy)
        self.info.l = self.cc.nearest(la.coords(cx, cy))
        self.info.c = self.cc.pointAt(self.info.l)
        pass
    def onRailInsertUpdate(self, ev):
        cx, cy = self.canvas.canvasxy(ev.x, ev.y)
        self.info.p = la.coords(cx, cy)
        self.info.l = self.cc.nearest(la.coords(cx, cy))
        self.info.c = self.cc.pointAt(self.info.l)

        if self.info.item:
            self.canvas.delete(self.info.item)
            self.info.item = None

        c = self.info.c
        p = self.info.p
        r = 12
        # bb = (c[0]-r, c[1]-r, c[0]+r, c[1]+r)
        bb = [c[0], c[1], p[0], p[1]]
        self.info.item = self.canvas.create_line(bb, fill="black")
    def onRailInsertEnd(self, ev):
        if self.info.item:
            self.canvas.delete(self.info.item)
            self.info.item = None
        # self.cr.addRoad(self.info.l,"NEW_STUFF")
        self.addHandles()
    def save(self):
        return self.cc
    def restore(self,data):
        self.cc = data
        self.addHandles()
###########################################################################
class CCManip(SavedFSM):
    def __init__(self, cc, canvas):
        self.cc = cc
        self.canvas = canvas

        self.segtag = "segment"
        self.movetag = "move"
        self.rottag = "rot"
        self.jointtag = "joint"

        s = Enum("States", "Idle Select Insert Append Move Rot SelRot SelScale Limbo")

        tt = {
            # (State, tag or None, event (None is any event)) -> (State, callback)

            # move control points, selection or segments
            (s.Idle, "move", "<ButtonPress-1>")      : (s.Move, self.onMoveStart),
            (s.Move, "move", "<B1-Motion>")          : (s.Move, self.onMoveUpdate),
            (s.Move, "move", "<ButtonRelease-1>")    : (s.Idle, self.onMoveEnd),

            (s.Idle, "joint", "<ButtonPress-1>")     : (s.Move, self.onJointMoveStart),
            (s.Move, "joint", "<B1-Motion>")         : (s.Move, self.onJointMoveUpdate),
            (s.Move, "joint", "<ButtonRelease-1>")   : (s.Idle, self.onJointMoveEnd),

            (s.Idle, "segment", "<ButtonPress-1>")   : (s.Move, self.onMoveStart),
            (s.Move, "segment", "<B1-Motion>")       : (s.Move, self.onMoveUpdate),
            (s.Move, "segment", "<ButtonRelease-1>") : (s.Idle, self.onMoveEnd),

            # rotate control point tangents
            (s.Idle, "rot", "<ButtonPress-1>")       : (s.Rot, self.onRotStart),
            (s.Rot, "rot", "<B1-Motion>")            : (s.Rot, self.onRotUpdate),
            (s.Rot, "rot", "<ButtonRelease-1>")      : (s.Idle, self.onRotEnd),

            # append point at end
            (s.Idle, None, "<ButtonPress-1>")        : (s.Limbo, None),
            (s.Limbo, None, "<B1-Motion>")           : (s.Append, self.onPointAppendStart),
            (s.Append, None, "<B1-Motion>")          : (s.Append, self.onPointAppendUpdate),
            (s.Append, None, "<ButtonRelease-1>")    : (s.Idle, self.onPointAppendEnd),

            # change type of segment
            (s.Idle, "segment", "<Button-2>")        : (s.Idle, self.onSegmentChangeType),

            # insert point in segment
            (s.Idle, "segment", "<Double-Button-1>") : (s.Limbo, self.onSegmentInsert),

            # remove point from segment
            (s.Idle, "move", "<Double-Button-1>")    : (s.Limbo, self.onPointRemove),

            (s.Idle, "joint", "<Double-Button-1>"): (s.Limbo, self.onSegmentSplitAtJoint),

            # extra state to consume button press and button release of double click
            (s.Limbo, None, "<ButtonRelease-1>")     : (s.Idle, None),

            # selection bindings
            (s.Idle, None, "<ButtonPress-3>")        : (s.Select, self.onSelectionStart),
            (s.Select, None, "<B3-Motion>")          : (s.Select, self.onSelectionUpdate),
            (s.Select, None, "<ButtonRelease-3>")    : (s.Idle, self.onSelectionEnd),

            (s.Idle, "move", "<Control-Button-3>")   : (s.Select, self.onSelectionToggle),
            (s.Select, "move", "<ButtonRelease-3>")  : (s.Idle, None),

            # alternative selection mode for non-3-button mouses
            (s.Idle, None, "<Key-s>")          : (s.Select, None),
            (s.Select, None, "<ButtonPress-1>")      : (s.Select, self.onSelectionStart),
            (s.Select, None, "<B1-Motion>")          : (s.Select, self.onSelectionUpdate),
            (s.Select, None, "<ButtonRelease-1>")    : (s.Select, self.onSelectionEnd),
            (s.Select, "move", "<Control-Button-1>") : (s.Select, self.onSelectionToggle),
            (s.Select, None, "<Key-s>")        : (s.Idle, None),

            # select all
            (s.Idle, None, "<Control-Key-a>")        : (s.Idle, self.onSelectAll),

            # selection rotation and scale bindings
            (s.Idle, None, "<Shift-ButtonPress-1>")  : (s.SelRot, self.onSelRotStart),
            (s.SelRot, None, "<Shift-B1-Motion>")    : (s.SelRot, self.onSelRotUpdate),
            (s.SelRot, None, "<ButtonRelease-1>")    : (s.Idle, self.onSelRotScaleEnd),

            (s.Idle, None, "<Control-Button-1>")     : (s.SelScale, self.onSelScaleStart),
            (s.SelScale, None, "<Control-B1-Motion>"): (s.SelScale, self.onSelScaleUpdate),
            (s.SelScale, None, "<ButtonRelease-1>")  : (s.Idle, self.onSelRotScaleEnd),

            (s.Idle, None, "<Key-l>")                : (s.Idle, self.onScaleLengthPopup),

            # toggle open bindings
            (s.Idle, None, "<Key-o>")                : (s.Idle, self.onToggleOpen),
            # reverse track
            (s.Idle, None, "<Key-r>")                : (s.Idle, self.onReverse),
        }
        super().__init__(s.Idle, tt, self.canvas)

        class EvInfo:
            canvas = self.canvas
            manip  = self
            def __init__(self,ev):
                self.cx, self.cy = EvInfo.canvas.canvasxy(ev.x, ev.y)
                self.dx, self.dy = 0,0
                self.px, self.py = self.cx, self.cy
                self.item = EvInfo.canvas.find_closest(self.cx, self.cy)[0]
                self.ox, self.oy = self.cx,self.cy
            def update(self,ev):
                self.cx, self.cy = EvInfo.canvas.canvasxy(ev.x, ev.y)
                self.dx, self.dy = self.cx - self.px, self.cy - self.py
                self.lx, self.ly = self.cx - self.ox, self.cy - self.oy
                self.px,self.py = self.cx, self.cy
                #self.item = EvInfo.canvas.find_closest(self.cx, self.cy)[0]
        self.EvInfo = EvInfo
        self.info = None

        self.cp_cidmap = {}
        self.seg_cidmap = {}
        self.imap = {}
        self.joint_cidmap = {}
        self.jointimap = {}
        self.selection = set()

        self.segstyle = {
            "width"  : 8,
            "outline": "#BEBEBE",
            "fill"   : "",
            "tags"   : self.segtag
        }
        self.movestyle = {
            "width"  : 8,
            "outline": "#B0C4DE",
            "fill"   : "",
            "tags"   : self.movetag
        }
        self.rotstyle = {
            "width"  : 8,
            "outline": "#EEDD82",
            "fill"   : "",
            "tags"   : self.rottag
        }
        self.jointstyle = {
            "width"  : 1,
            "outline": "#EE82DD",
            "fill"   : "#EE82DD",
            "tags"   : self.jointtag
        }

        style_active(self.segstyle, -0.1, 0, 2)
        style_active(self.movestyle, -0.1, 0.1, 3)
        style_active(self.rotstyle, -0.1, 0.1, 3)
        style_active(self.jointstyle, -0.1, 0.1, 24)

        self.selstyle = style_modified(self.movestyle, -0.2, 0.2, 2)

        self.lengthdisplay = self.canvas.create_text(self.canvas.canvasxy(10, 10),
                                                     text="Length: {:.2f}m".format(self.cc.length()),
                                                     anchor=tk.NW,
                                                     tags="fixed-to-window")

        self.redrawSegments()
        self.addHandles()

    def redrawSegments(self, affected=None, except_seg=None):
        if except_seg is None: except_seg = []
        if affected is None:
            for seg, cids in self.jointimap.items():
                for cid in cids:
                    self.canvas.delete(cid)
            # TODO: remove nonexisting segments from imap
            self.canvas.delete(self.segtag)

            seg2cid = self.cc.draw(self.canvas, **self.segstyle)
            self.seg_cidmap = {}
            for s, cids in seg2cid.items():
                self.imap[s] = cids
                for cid in cids:
                    self.seg_cidmap[cid] = s
        else:
            for i, a in enumerate(affected):
                if a in self.imap:
                    cids = self.imap[a]
                    for cid in cids:
                        self.canvas.delete(cid)
                ncids = self.cc.drawSegment(a, self.canvas, **self.segstyle)
                if ncids is None:
                    self.imap.pop(a)
                    for cid in cids:
                        self.seg_cidmap.pop(cid)
                else:
                    self.imap[a] = ncids
                    for cid in ncids:
                        self.seg_cidmap[cid] = a
        self.addJointHandles(except_seg)
        self.canvas.tag_raise(self.segtag, "contour")
        try:
            self.canvas.tag_raise(self.segtag, "image")
        except tk.TclError:
            pass

        # minslotlen = 50
        # tex = m.fmod(self.cc.length(),minslotlen)
        # numslots = int((self.cc.length() - tex)/minslotlen + 2)
        # ex = (self.cc.length() - numslots*minslotlen)/numslots
        # slotlen = minslotlen + ex
        
        # try:
        #     self.canvas.delete("slots")
        # except tk.TclError:
        #     pass

        # for i in range(numslots):
        #     s = i*slotlen
        #     op, ot = self.cc.pointAndTangentAt(s)
        #     on = la.perp2ccw(ot)
        #     p1 = op + 30*la.unit(on)
        #     p2 = op - 30*la.unit(on)
        #     self.canvas.create_polygon([(p1[0],p1[1]),(p2[0],p2[1])],outline="black",tags="slots")

        self.canvas.itemconfigure(self.lengthdisplay,
                                  text="Length: {:.2f}m".format(self.cc.length()))
    def addJointHandles(self, except_seg=None):
        if except_seg is None: except_seg = []
        # redraw all joint handles except for segments in except_seg
        for seg, cids in self.jointimap.items():
            if seg in except_seg:
                continue
            for cid in cids:
                self.canvas.delete(cid)  # remove joint handles
        for seg in self.cc.segment:
            if seg in except_seg:
                continue
            if seg.type is SegType.Biarc:
                cids = [self.addJointHandle(seg)]
                for cid in cids:
                    self.joint_cidmap[cid] = seg
                    if seg in self.jointimap:
                        self.jointimap[seg].extend(cids)
                    else:
                        self.jointimap[seg] = cids

    def addRotHandle(self, cp):
        c = cp.point
        t = cp.tangent
        p = c + 35 * t
        r = 12
        cid = canvas_create_circle(self.canvas, p, r, **self.rotstyle)
        return cid
    def addMoveHandle(self, cp):
        c = cp.point
        r = 12
        cid = canvas_create_circle(self.canvas, c, r, **self.movestyle)
        return cid
    def addJointHandle(self, seg):
        c = seg.seg.joint()
        r = 3
        cid = canvas_create_circle(self.canvas, c, r, **self.jointstyle)
        return cid
    def removeHandles(self):
        self.canvas.delete(self.movetag)
        self.canvas.delete(self.rottag)
        self.canvas.delete(self.jointtag)
    def addHandles(self):
        self.removeHandles()
        self.cp_cidmap = {}
        for cp in self.cc.point:
            cids = []
            cid1 = self.addMoveHandle(cp)
            self.cp_cidmap[cid1] = cp
            cids.append(cid1)
            if self.cc.tangentUpdateable(cp):
                cid2 = self.addRotHandle(cp)
                self.cp_cidmap[cid2] = cp
                cids.append(cid2)
            self.imap[cp] = cids
        self.addJointHandles()

    def onSegmentChangeType(self, ev):
        self.historySave()
        self.info = self.EvInfo(ev)
        seg = self.seg_cidmap[self.info.item]
        aff = self.cc.changeType(seg)
        self.redrawSegments(aff)
        self.addHandles()
    def onReverse(self, ev):
        self.historySave()
        aff = self.cc.reverse()
        self.redrawSegments(aff)
        self.addHandles()
    def onPointAppendStart(self, ev):
        if not self.cc.isOpen: return
        self.info = self.EvInfo(ev)
        self.historySave()
        pos = la.coords(self.info.cx, self.info.cy)
        self.info.cp, self.info.seg, self.info.aff = self.cc.appendPoint(pos, SegType.Biarc)
        self.cc.setTangent(self.info.cp, None)
        self.redrawSegments([self.info.seg])
    def onPointAppendUpdate(self, ev):
        if not self.cc.isOpen: return
        self.info.update(ev)
        self.cc.movePoint(self.info.cp, la.coords(self.info.dx, self.info.dy))
        self.cc.setTangent(self.info.cp, None)
        self.redrawSegments([self.info.seg])
    def onPointAppendEnd(self, ev):
        if not self.cc.isOpen: return
        self.info = None
        self.addHandles()
    def onSegmentInsert(self, ev):
        self.historySave()
        self.info = self.EvInfo(ev)
        seg = self.seg_cidmap[self.info.item]
        pos = la.coords(self.info.cx, self.info.cy)
        cp, seg2, aff = self.cc.insertPoint(seg, pos, SegType.Biarc)
        self.redrawSegments(aff)
        self.addHandles()
    def onSegmentSplitAtJoint(self, ev):
        self.historySave()
        self.info = self.EvInfo(ev)
        seg = self.joint_cidmap[self.info.item]
        pos = seg.seg.joint()
        cp, seg2, aff = self.cc.insertPoint(seg, pos, SegType.Biarc)
        self.redrawSegments(aff)
        self.addHandles()
    def onPointRemove(self, ev):
        self.historySave()
        self.info = self.EvInfo(ev)
        cp = self.cp_cidmap[self.info.item]
        aff = self.cc.removePoint(cp)
        self.redrawSegments(aff)
        self.addHandles()
    def onSelectionStart(self, ev):
        self.addHandles()
        self.selection.clear()
        self.info = self.EvInfo(ev)
        self.info.selstart = la.coords(self.info.cx, self.info.cy)
        self.info.selend = None
        self.info.selcid = None
    def redrawSelection(self):
        if self.info and hasattr(self.info,"selend"):
            selpoly = [self.info.selstart[0], self.info.selstart[1],
                       self.info.selstart[0], self.info.selend[1],
                       self.info.selend[0], self.info.selend[1],
                       self.info.selend[0], self.info.selstart[1]]
            if self.info.selcid:
                self.canvas.delete(self.info.selcid)

            self.info.selcid = self.canvas.create_polygon(selpoly, fill="", outline="grey")

        for cid in self.canvas.find_withtag(self.movetag):
            self.canvas.itemconfig(cid, self.movestyle)

        for cp in self.selection:
            cids = self.imap[cp]
            for cid in cids:
                if self.rottag in self.canvas.gettags(cid):  # only modify the move handles
                    continue
                self.canvas.itemconfig(cid, self.selstyle)

    def onSelectionUpdate(self, ev):
        self.info.update(ev)
        self.info.selend = la.coords(self.info.cx, self.info.cy)
        cids = self.canvas.find_overlapping(self.info.selstart[0], self.info.selstart[1],
                                            self.info.selend[0], self.info.selend[1])
        self.selection.clear()
        for cid in cids:
            if cid not in self.cp_cidmap:
                continue
            cp = self.cp_cidmap[cid]
            self.selection.add(cp)

        self.redrawSelection()
    def onSelectionEnd(self, ev):
        if self.info:
            self.canvas.delete(self.info.selcid)
        self.info = None
    def onSelectionToggle(self, ev):
        self.info = self.EvInfo(ev)
        cp = self.cp_cidmap[self.info.item]
        if cp in self.selection:
            self.selection.remove(cp)
        else:
            self.selection.add(cp)
        self.redrawSelection()
    def onSelectAll(self, ev):
        if len(self.selection) == len(self.cc.point):
            self.selection.clear()
        else:
            self.selection = set(self.cc.point)
        self.redrawSelection()
    def onScaleLengthPopup(self, ev):
        # print("onScaleLengthPopup")
        text = """
    Enter new track length:
    """
        self.pe = PopupEntry(self.canvas.master, (ev.x, ev.y),
                             text,
                             self.onScaleLengthPopupDone,
                             "Scale to Length")
        # sys.stdout.flush()
    def onScaleLengthPopupDone(self, length):
        self.historySave()
        # print("onScaleLengthPopupDone",length)
        desired_length = float(length)
        scale = desired_length / self.cc.length()
        bbox = self.canvas.bbox("segment")
        mx = abs(bbox[0] - bbox[2])
        my = abs(bbox[1] - bbox[3])
        scale_origin = (mx, my)  # point that stays fixed while scaling
        xform = la.identity()
        xform = la.mul(xform, la.translate(scale_origin[0], scale_origin[1]))
        xform = la.mul(xform, la.scale(scale, scale, scale))
        xform = la.mul(xform, la.translate(-scale_origin[0], -scale_origin[1]))
        self.applySelXForm(xform)
    def onMoveStart(self, ev):
        self.info = self.EvInfo(ev)
        self.info.sel = self.selection
        self.info.seg = None
        self.info.mod = False
        if self.info.item in self.cp_cidmap:
            cp = self.cp_cidmap[self.info.item]
            if not self.selection or cp not in self.info.sel:
                self.info.sel = [cp]
        elif self.info.item in self.seg_cidmap:
            seg = self.seg_cidmap[self.info.item]
            self.info.sel = self.cc.fixedSegmentPoints(seg)
            self.info.seg = seg
    def onMoveUpdate(self, ev):
        self.info.update(ev)
        if not self.info.mod:
            self.historySave()
        self.info.mod = True

        xform = la.translate(self.info.dx, self.info.dy)
        aff = self.cc.transformPoints(self.info.sel, xform)
        #cids = []
        #for s in xformable:
        #    cids.extend(self.imap[s])
        #self.canvas.apply_xform(cids, xform)

        for cp in self.info.sel:
            cids = self.imap[cp]
            for cid in cids:
                self.canvas.move(cid, self.info.dx, self.info.dy)

        if self.info.seg:
            aff.remove(self.info.seg)
            cids = self.imap[self.info.seg]
            for cid in cids:
                self.canvas.move(cid, self.info.dx, self.info.dy)

        self.redrawSegments(aff)
    def onMoveEnd(self, ev):
        self.info = None
    def onJointMoveStart(self, ev):
        self.info = self.EvInfo(ev)
        self.info.seg = self.joint_cidmap[self.info.item]
        self.info.mod = False
    def onJointMoveUpdate(self, ev):
        self.info.update(ev)
        if not self.info.mod:
            self.historySave()
        self.info.mod = True

        aff, dx, dy = self.cc.moveJoint(self.info.seg, la.coords(self.info.cx, self.info.cy))
        cids = self.jointimap[self.info.seg]
        for cid in cids:
            self.canvas.move(cid, dx, dy)
        self.redrawSegments(aff, aff)
    def onJointMoveEnd(self, ev):
        self.info = None
    def onRotStart(self, ev):
        self.info = self.EvInfo(ev)
        self.info.mod = False
    def onRotUpdate(self, ev):
        self.info.update(ev)
        if not self.info.mod:
            self.historySave()
        self.info.mod = True

        cp = self.cp_cidmap[self.info.item]
        p = cp.point
        ot = cp.tangent
        pos = la.coords(self.info.cx, self.info.cy)
        t, l = la.unit_length(pos - p)

        d = 35 * t - 35 * ot
        da = 70 * t - 70 * ot

        aff = self.cc.setTangent(cp, t)

        self.canvas.move(self.info.item, d[0], d[1])

        self.redrawSegments(aff)
    def onRotEnd(self, ev):
        self.info = None
    def applySelXForm(self, xform):
        aff = self.cc.transformPoints(self.selection, xform)
        #cids = []
        #for s in xformable:
        #    cids.extend(self.imap[s])
        #self.canvas.apply_xform(cids, xform)
        self.addHandles()
        self.redrawSegments(aff)
        self.redrawSelection()
    def onSelRotStart(self, ev):
        if not self.selection: return
        self.info = self.EvInfo(ev)
        self.info.preva = None

        c = la.coords(self.info.cx, self.info.cy)
        r = 40
        self.info.center = canvas_create_circle(self.canvas, c, r)
    def onSelRotUpdate(self, ev):
        if not self.selection: return
        self.info.update(ev)

        t, l = la.unit_length(la.coords(self.info.lx, self.info.ly))
        cura = m.atan2(t[1], t[0])

        if self.info.preva is None:
            self.historySave()
        elif l < 40:  # don't rotate too near on center
            pass
        else:
            xform = la.identity()
            xform = la.mul(xform, la.translate(self.info.ox, self.info.oy))
            a = cura - self.info.preva
            xform = la.mul(xform, la.rotate(a, 0, 0, 1))
            xform = la.mul(xform, la.translate(-self.info.ox, -self.info.oy))
            self.applySelXForm(xform)
        self.info.preva = cura
        print(self.info.preva)
        sys.stdout.flush()
    def onSelScaleStart(self, ev):
        if not self.selection: return
        self.info = self.EvInfo(ev)
        self.info.preva = None

        c = la.coords(self.info.cx, self.info.cy)
        r = 60
        self.info.center = canvas_create_circle(self.canvas, c, r)

    def onSelScaleUpdate(self, ev):
        if not self.selection: return
        self.info.update(ev)

        t, l = la.unit_length(la.coords(self.info.lx, self.info.ly))

        cura = l

        if self.info.preva is None:
            if l < 60:
                return
            self.historySave()
        elif l < 10:  # don't scale too near on center
            return
        else:
            xform = la.identity()
            xform = la.mul(xform, la.translate(self.info.ox, self.info.oy))
            a = cura / self.info.preva
            xform = la.mul(xform, la.scale(a, a, a))
            xform = la.mul(xform, la.translate(-self.info.ox, -self.info.oy))
            self.applySelXForm(xform)
        self.info.preva = cura
        print(self.info.preva)
        sys.stdout.flush()
    def onSelRotScaleEnd(self, ev):
        if self.info:
            self.canvas.delete(self.info.center)
        self.info = None
    def onToggleOpen(self, ev, *args):
        self.historySave()
        aff = self.cc.toggleOpen(*args)
        self.redrawSegments([aff])
        self.addHandles()
    def save(self):
        return self.cc
    def restore(self,data):
        self.cc = data
        self.selection.clear()  # clear selection
        self.redrawSegments()
        self.addHandles()
###########################################################################
class BankingManip(tk.Frame,SavedFSM):
    def __init__(self, app, cc, master=None):
        tk.Frame.__init__(self,master)
        self.app = app
        self.cc = cc
        self.pack()

        # setup frame
        self.canvas = CX.CanvasX(self, width=800, height=380)
        self.hbar = tk.Scrollbar(self, orient=tk.HORIZONTAL)
        self.hbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.hbar.config(command=self.canvas.xview)
        self.vbar = tk.Scrollbar(self, orient=tk.VERTICAL)
        self.vbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.vbar.config(command=self.canvas.yview)

        self.canvas.config(scrollregion=(-10, -190, self.cc.length() + 10, 190), confine=True)
        self.canvas.config(xscrollcommand=self.hbar.set, yscrollcommand=self.vbar.set)

        self.canvas.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)
        self.canvas.focus_set()

        self.imap = {}
        self.textcid = None

        self.drawCoords()
        self.drawBanking()
        self.drawTransition(True)
        # setup state machine

        s = Enum("States", "Idle Bank Trans")
        tt = {
            # (State, tag or None, event (None is any event)) -> (State, callback)
            (s.Idle, None, "<Motion>")           : (s.Idle, self.onMouseMotion),
            (s.Idle, None, "<MouseWheel>")       : (s.Idle, self.onWheel),
            (s.Idle, None, "<Button-4>")         : (s.Idle, self.onWheel),
            (s.Idle, None, "<Button-5>")         : (s.Idle, self.onWheel),
            (s.Idle, None, "<Configure>")        : (s.Idle, self.onConfigure),

            (s.Idle, "bank", "<ButtonPress-1>")  : (s.Bank, self.onMoveStart),
            (s.Bank, "bank", "<B1-Motion>")      : (s.Bank, self.onBankMove),
            (s.Bank, "bank", "<ButtonRelease-1>"): (s.Idle, self.onMoveEnd),
            (s.Idle,  "transpoint", "<ButtonPress-1>")   : (s.Trans, self.onMoveStart),
            (s.Trans, "transpoint", "<B1-Motion>")       : (s.Trans, self.onTransMove),
            (s.Trans, "transpoint", "<ButtonRelease-1>") : (s.Idle,  self.onMoveEnd),
        }
        SavedFSM.__init__(self,s.Idle, tt, self.canvas)
        self.mod = None

    def onConfigure(self, ev):
        self.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)
        self.canvas.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)
    def onWheel(self, ev):
        cx, cy = self.canvas.canvasxy(ev.x, ev.y)
        sf = 1.1
        if not (ev.delta == 0):  # Wheel event
            if ev.delta < 0: sf = 1 / sf
        else:
            if ev.num == 5: sf = 1 / sf
        # scale all objects on canvas
        self.canvas.zoom(cx, cy, sf)
    def onMouseMotion(self, ev):
        l, a = self.canvas.canvasxy(ev.x, ev.y)
        if l < 0: l = 0
        if l > self.cc.length(): l = self.cc.length()
        self.app.drawTrackIndicator(l)
        if self.textcid:
            self.canvas.delete(self.textcid)
        self.textcid = self.canvas.create_text(self.canvas.canvasxy(10, 10),
                                               text="{:.2f}m\n{:.4f}°".format(l, a),
                                               anchor=tk.NW, tag="fixed-to-window")

    def drawCoords(self):
        self.canvas.create_line(0, 0, self.cc.length(), 0, fill="black", tag="grid")
        self.canvas.create_line(0, -180, 0, 180, fill="black", tag="grid")
        for i in range(10, 180, 10):
            self.canvas.create_line(0, +i, self.cc.length(), +i, fill="grey", tag="grid")
            self.canvas.create_line(0, -i, self.cc.length(), -i, fill="grey", tag="grid")
        self.canvas.tag_lower("grid")
    def drawTransition(self, withPoints=False):
        self.canvas.delete("trans")
        if withPoints:
            delitem = []
            for cid, bk in self.imap.items():
                if "transpoint" in self.canvas.gettags(cid):
                    delitem.append(cid)
            for cid in delitem:
                self.canvas.delete(cid)
                self.imap.pop(cid)
        tl = 0
        for i, seg in enumerate(self.cc.segment):
            prev = self.cc.segment[i - 1]
            next = self.cc.segment[i + 1 if i < len(self.cc.segment) - 1 else 0]
            shapes_p = prev.shapeparameters()
            shapes = seg.shapeparameters()
            ns = len(shapes)

            bk_p = (prev.banking[len(shapes_p) - 1], seg.banking[0])
            bk_n = (next.banking[0], seg.banking[ns - 1])

            for j, shape in enumerate(shapes):
                ep, l, div, k, center = shape
                bk = seg.banking[j]
                bk1 = bk_p[j]
                bk2 = bk_n[ns - 1 - j]

                bkm1 = (bk1.angle + bk.angle) / 2
                bkm2 = (bk2.angle + bk.angle) / 2
                self.canvas.create_line(tl, bkm1, tl + bk.prev_len, bk.angle,
                                        width=1, fill="lightblue", tags="trans")
                self.canvas.create_line(tl + l - bk.next_len, bk.angle, tl + l, bkm2,
                                        width=1, fill="lightblue", tags="trans")
                if withPoints:
                    transcid = canvas_create_circle(self.canvas, [tl + bk.prev_len, bk.angle], 2,
                                                    width=1, activewidth=3, fill="blue", tags="transpoint")
                    self.imap[transcid] = (bk, "prev", tl, tl + l)
                    transcid = canvas_create_circle(self.canvas, [tl + l - bk.next_len, bk.angle], 2,
                                                    width=1, activewidth=3, fill="blue", tags="transpoint")
                    self.imap[transcid] = (bk, "next", tl, tl + l)
                tl += l
    def drawBanking(self):
        delitem = []
        for cid, bk in self.imap.items():
            if "bank" in self.canvas.gettags(cid):
                delitem.append(cid)
        for cid in delitem:
            self.canvas.delete(cid)
            self.imap.pop(cid)
        tl = 0
        for i, seg in enumerate(self.cc.segment):
            shapes = seg.shapeparameters()
            for j, shape in enumerate(shapes):
                ep, l, div, k, center = shape
                bk = seg.banking[j]
                bankcid = self.canvas.create_line(tl + bk.prev_len, bk.angle, tl + l - bk.next_len, bk.angle,
                                                  width=3, activewidth=5,
                                                  fill="blue", tags="bank")
                self.imap[bankcid] = bk
                tl += l
    def save(self):
        return self.cc
    def restore(self,data):
        self.cc = data
        self.drawBanking()
        self.drawTransition(True)
    def onMoveStart(self,ev):
        self.mod = False
    def onMoveEnd(self,ev):
        self.mod = False
    def onTransMove(self, ev):
        if self.mod is False:
            self.historySave()
        self.mod = True
        l, a = self.canvas.canvasxy(ev.x, ev.y)
        cid = self.canvas.find_withtag("current")[0]
        bk, prev_or_next, bl, el = self.imap[cid]
        dl = 0
        if l < bl:
            l = bl
        if l > el:
            l = el
        if prev_or_next == "prev":
            bk.prev_len = l - bl
        else:
            bk.next_len = el - l
        sl, sa, el, ea = self.canvas.coords(cid)
        self.canvas.move(cid, l - (sl + el) / 2, 0)
        self.drawTransition()
        self.drawBanking()
    def onBankMove(self, ev):
        if self.mod is False:
            self.historySave()
        self.mod = True
        l, a = self.canvas.canvasxy(ev.x, ev.y)
        cid = self.canvas.find_withtag("current")[0]
        bk = self.imap[cid]
        bk.angle = a
        sl, sa, el, ea = self.canvas.coords(cid)
        self.canvas.move(cid, 0, a - sa)
        self.drawTransition(True)
###########################################################################
class App(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.pack()
        self.setup()
    def askOpenFileName(self):
        path = filedialog.askopenfilename()
        print(path)
        return path
    def askSaveFileName(self):
        path = filedialog.asksaveasfilename()
        print(path)
        return path
    def loadCP(self):
        path = self.askOpenFileName()
        try:
            self.cc = pickle.load(open(path, "rb"))
            self.ccmanip.cc = self.cc
            self.ccmanip.redrawSegments()
            self.ccmanip.addHandles()
            self.recenterTrack()
        except FileNotFoundError:
            print("file not found!")
            # self.ccmanip = CCManip(self.cc,self.canvas)
    def saveCP(self):
        path = self.askSaveFileName()
        try:
            pickle.dump(self.cc, open(path, "wb"))
        except FileNotFoundError:
            print("file not found!")
    def loadAndImportTed(self):
        path = self.askOpenFileName()
        tedfile = ""
        try:
            with open(path, mode='rb') as file:
                tedfile = file.read()
        except FileNotFoundError:
            print("file not found!")
            return
        self.importTed(tedfile)

        for name, data in the_scenery.items():
            if data["id"] == self.cc.scenery:
                self.switchScenery(name)

        # rebuild
        self.railmanip.restore(self.cc)
        self.ccmanip.restore(self.cc)
        #self.ccmanip.redrawSegments()
        #self.ccmanip.addHandles()

        self.recenterTrack()
    def importTed(self,tedfile):
        return self.cc.importTed(tedfile)
    def exportTed(self):
        return self.cc.exportTed()
    def saveTedToFile(self, teddata):
        # write to disk
        path = self.askSaveFileName()
        try:
            with open(path, mode='wb') as file:
                file.write(teddata)
        except FileNotFoundError:
            print("file not found!")
    def exportAndSaveTed(self):
        teddata = self.exportTed()
        self.saveTedToFile(teddata)
    def onAbout(self):
        text = """
    The Ted Editor

    by tarnheld with the help of the GTPlanet community

    Special Thanks to eran0004, MrGrumpy, NingDynasty,
    Outspacer, PR1VATEJ0KER, Patrick8308, Razerman and all i forgot!

    includes the Elevation Editor by eran0004, uploadTed code by Razerman

    """
        self.about = PopupAbout(self.canvas.master, (10, 10), text)
    def onDisclaimer(self):
        text = """
    DISCLAIMER

    By using the Upload TED command, you will possibly violate the
    Playstation Network Terms of use, and that might result in your
    account being banned! While no one so far has been banned by
    uploading track with software other than the official GT6 Track
    Path Editor, there have been bans because of modding cars and time
    trial entries. The Ted Editor including the track upload feature
    is provided for adding awesome new tracks with features not
    possible before to GT6, not for gaining advantage over other
    players, use it wisely."""
        self.disclaimer = PopupAbout(self.canvas.master, (10, 10), text)

    def uploadTed(self):
        if not self.disclaimer:
            self.onDisclaimer()
        if not self.tedfile:
            self.tedfile = self.exportTed()
        if not self.cookie:
            text = """
      Enter the gran-turismo.com session id, to find it log in at the
      community area at www.gran-turismo.com and copy the session id
      from the cookies of this domain:

      * in Firefox:
      -> right click on site content
      -> show site information
      -> Security tab
      -> Show Cookies

      * in Chrome:
      -> click on the security information (small lock) left of the addressbar
      -> Cookies

      triple click on the content string to select it completely (looks like
      xxxxxxxxxx_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx.workerxx)
      and paste it here:

      """
            self.pe = PopupEntry(self.canvas.master, (10, 10), text, self.uploadTedWithCookie, "Login Cookie")
        else:
            self.uploadTedWithCookie(self.cookie)
    def uploadTedWithCookie(self, cookie, title="track from from ted editor", country="de"):
        username = ut.checkCookieValidity(cookie)
        data = ut.checkTedDataValidity(self.tedfile)
        self.cookie = cookie  # save cookie for later
        if username and data and ut.uploadTedData(data, title, cookie, username):
            print("upload successful!")
        elif not username:
            print("cookie invalid, please log in again")
            self.cookie = None  # invalidate cached cookie
        elif not data:
            print("ted file invalid")
        else:
            print("upload failed, maybe more than 30 tracks used already?")

        # import requests
        # import json
        # headers = {'Cookie': 'JSESSIONID='+cookie}
        # print(headers)
        # r = requests.post('http://www.gran-turismo.com/gb/ajax/get_online_id/', headers=headers)
        # print(r.text)
        # username = json.loads(r.text)["online_id"]
        # if not username:
        #     # Print error message, cookie is invalid
        #     print("Error: Cookie is invalid!")
        #     self.cookie = None
        #     return False
        # else:
        #     files = {"data": ("gt6.ted", self.tedfile)}
        #     data = {'job': (None, '1'), 'user_id': (None, username), 'title': (None, title)}
        #     res = requests.post('https://www.gran-turismo.com/'+country+'/api/gt6/course/', files=files, data=data, headers=headers, verify=False)
        #     uploadResult = json.loads(res.text)["result"]
        #     if uploadResult == 1:
        #         print("Upload succeeded!")
        #         return True
        #     else:
        #         print("Upload failed! Could be due to 30 tracks limit?")
        #         return False
        sys.stdout.flush()
    def recenterTrack(self):
        bbox = self.canvas.bbox("segment")

        ctx, cty = self.canvas.canvasxy(0, 0)
        cbx, cby = self.canvas.canvasxy(self.canvas.winfo_width(), self.canvas.winfo_height())
        eox = abs(ctx - cbx)
        eoy = abs(cty - cby)
        enx = abs(bbox[0] - bbox[2])
        eny = abs(bbox[1] - bbox[3])
        aro = eox / eoy
        arn = enx / eny

        if aro < arn:
            sf = eox / enx
        else:
            sf = eoy / eny

        self.canvas.zoom((ctx + cbx) / 2, (cty + cby) / 2, 1 / 1000)
        self.canvas.zoom((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2, 1000 * 0.9 * sf)
    def clearTrackIndicator(self):
        if self.ti_cid:
            for cid in self.ti_cid:
                self.canvas.delete(cid)
            self.ti_cid = None
    def drawTrackIndicator(self, l):
        self.clearTrackIndicator()
        p = self.cc.pointAt(l)
        self.ti_cid = [canvas_create_circle(self.canvas, p, 10)]
    def drawTrackOffsetIndicator(self, ls, le, f0, f1):
        self.clearTrackIndicator()
        bpoly = self.cc.offsetPolygonAt(ls, le, 20)
        f0poly = self.cc.offsetPolygonAt(ls - f0, ls, 20)
        f1poly = self.cc.offsetPolygonAt(le, le + f1, 20)
        self.ti_cid = [
            self.canvas.create_polygon([(x[0], x[1]) for x in bpoly], fill="", outline="blue"),
            self.canvas.create_polygon([(x[0], x[1]) for x in f0poly], fill="", outline="lightblue"),
            self.canvas.create_polygon([(x[0], x[1]) for x in f1poly], fill="", outline="lightblue")
        ]
        # sys.stdout.flush()

    def switchScenery(self, name):
        global the_heightmap
        print("switch scenery", name)

        self.scene = the_scenery[name]

        the_heightmap = Heightmap(self.scene["npz"],self.scene["ex"])

        self.cc.scenery = self.scene["id"]
        
        self.drawContours(self.scene["ct"])
        #if hasattr(self,"railmanip"):
        #    self.railmanip.switchScenery(self.scene)

    def setup(self):

        # create a toplevel menu
        self.menubar = tk.Menu(self)

        filemenu = tk.Menu(self.menubar, tearoff=0)
        filemenu.add_command(label="Load Track", command=self.loadCP)
        filemenu.add_command(label="Save Track", command=self.saveCP)
        filemenu.add_command(label="Import TED", command=self.loadAndImportTed)
        filemenu.add_command(label="Export TED", command=self.exportAndSaveTed)
        filemenu.add_separator()
        filemenu.add_command(label="Upload TED", command=self.uploadTed)
        filemenu.add_separator()
        filemenu.add_command(label="Import Image", command=self.importImg)
        filemenu.add_command(label="Discard Image", command=self.discardImg)
        filemenu.add_separator()
        filemenu.add_command(label="Quit", command=self.quit)
        self.menubar.add_cascade(label="File", menu=filemenu)
        helpmenu = tk.Menu(self.menubar, tearoff=0)
        helpmenu.add_command(label="About", command=self.onAbout)
        helpmenu.add_command(label="Upload Disclaimer", command=self.onDisclaimer)
        self.menubar.add_cascade(label="Help", menu=helpmenu)

        scenerymenu = tk.Menu(self.menubar, tearoff=0)
        for name, data in the_scenery.items():
            print("add scenery menu", name)
            scenerymenu.add_command(label=name, command=lambda s=self, n=name: s.switchScenery(n))
        self.menubar.add_cascade(label="Scenery", menu=scenerymenu)

        self.master.config(menu=self.menubar)

        self.canvas = CX.CanvasX(self, width=800, height=600)
        self.hbar = tk.Scrollbar(self, orient=tk.HORIZONTAL)
        self.hbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.hbar.config(command=self.canvas.xview)
        self.vbar = tk.Scrollbar(self, orient=tk.VERTICAL)
        self.vbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.vbar.config(command=self.canvas.yview)
        self.extent = 10000

        self.canvas.config(scrollregion=(-self.extent, -self.extent, self.extent, self.extent), confine=True)
        self.canvas.config(xscrollcommand=self.hbar.set, yscrollcommand=self.vbar.set)

        self.canvas.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)

        self.canvas.bind("<ButtonPress-2>", self.onDragStart)
        self.canvas.bind("<ButtonRelease-2>", self.onDragEnd)
        self.canvas.bind("<Shift-Key-d>", self.onDragToggle)
        self.canvas.bind("<MouseWheel>", self.onWheel)
        self.canvas.bind("<Button-4>", self.onWheel)
        self.canvas.bind("<Button-5>", self.onWheel)
        self.canvas.bind("<Motion>", self.onDrag)
        self.canvas.bind("<Key-m>", self.onToggleManip)
        self.canvas.bind("<Key-b>", self.onBankingEdit)
        self.canvas.bind("<Key-e>", self.onElevationEdit)

        self.canvas.bind("<Configure>", self.onConfigure)
        self.canvas.focus_set()
        self.drawCoordGrid()

        self.road_width = 8
        self.cc = ControlCurve(self.road_width)

        self.switchScenery("Andalusia")

        self.img = None
        self.simg = None
        self.pimg = None
        self.imgcid = None
        self.bankmanip = None
        self.bankwindow = None
        self.eleveditor = None
        self.elevwindow = None
        self.ti_cid = None

        self.cookie = None
        self.tedfile = None

        self.disclaimer = None
        self.about = None

        self.dragging = False

        # initial simple track
        self.cc.appendPoint(la.coords(100, 200))
        self.cc.appendPoint(la.coords(100, 300))
        self.cc.appendPoint(la.coords(200, 450), SegType.Biarc)
        self.cc.appendPoint(la.coords(600, 450))
        self.cc.appendPoint(la.coords(700, 300), SegType.Biarc)
        self.cc.appendPoint(la.coords(600, 200), SegType.Biarc)
        self.cc.appendPoint(la.coords(400, 300))
        self.cc.toggleOpen()
        
        self.ccmanip = CCManip(self.cc, self.canvas)
        self.ccmanip.stop()
        self.railmanip = RailManip(self.cc, self.canvas)
        self.railmanip.switchScenery(self.scene)
        self.railmanip.stop()
        self.ccmanip.start()


    def onToggleManip(self, ev):
        if self.ccmanip.isStopped():
            self.railmanip.stop()
            self.railmanip.removeHandles()
            self.ccmanip.restore(self.railmanip.save())
            self.ccmanip.start()
        else:
            self.ccmanip.stop()
            self.ccmanip.removeHandles()
            self.railmanip.restore(self.ccmanip.save())
            self.railmanip.start()
    def restoreManip(self, cc):
        if self.ccmanip.isStopped():
            self.railmanip.restore(cc)
        else:
            self.ccmanip.restore(cc)
    def onBankingEdit(self, ev):
        if not self.bankwindow:
            self.bankwindow = tk.Toplevel(self.master)
            self.bankmanip = BankingManip(self, self.cc, self.bankwindow)
            self.bankwindow.protocol("WM_DELETE_WINDOW", self.onBankingClose)
            self.ccmanip.stop()
            self.railmanip.stop()
            self.ccmanip.removeHandles()
            self.railmanip.removeHandles()
            if self.ccmanip.isStopped():
                self.bankmanip.restore(self.railmanip.save())
            else:
                self.bankmanip.restore(self.ccmanip.save())
        else:
            self.onBankingClose()
    def onBankingClose(self):
        self.ccmanip.restore(self.bankmanip.save())
        self.bankmanip = None
        self.bankwindow.destroy()
        self.bankwindow = None
        self.clearTrackIndicator()
        self.ccmanip.addHandles()
        self.ccmanip.start()

    def onElevationEdit(self, ev):
        if not self.elevwindow:
            self.elevwindow = tk.Toplevel(self.master)
            self.eleveditor = eed.ElevationEditor(self.elevwindow, self)
            self.tedfile = self.exportTed()
            trackobj = eed_rf2.initFromTedFile(self.tedfile)
            self.eleveditor._load_ted(trackobj)
            self.elevwindow.protocol("WM_DELETE_WINDOW", self.onElevationClose)
            self.ccmanip.stop()
            #self.ccmanip.removeHandles()
            self.railmanip.stop()
            #self.railmanip.removeHandles()
        else:
            self.onElevationClose()
    def onElevationClose(self):
        tokenlist = self.eleveditor._gen_tokenlist()
        self.eleveditor.structure.mod = [token[1] + self.eleveditor._first_z for token in tokenlist]
        tedfile = eed_rf2.generateTedFile(self.eleveditor.structure)
        self.importTed(tedfile)

        self.eleveditor = None
        self.elevwindow.destroy()
        self.elevwindow = None
        self.clearTrackIndicator()
        self.ccmanip.restore(self.cc)
        self.ccmanip.addHandles()
        self.ccmanip.start()

    def drawContours(self, path):
        # print("load contours",path)
        try:
            self.canvas.delete("contour")
        except tk.TclError:
            pass

        import re
        self.contours = np.load(path)
        ex = self.scene["ex"]
        lf = len(self.contours.files)
        for a in self.contours.files:
            cs = self.contours[a]
            h = int(re.findall('\d+', a)[0])
            h /= lf
            # print("file",a,h)
            # print("contours",len(cs))
            col = colorsys.rgb_to_hsv(0.7, 0.9, 0.85)
            hue = col[0] - h / 2
            hue = m.fmod(hue, 1)
            col = (hue, max(0, min(col[1], 1)), max(0, min(col[2], 1)))
            col = colorsys.hsv_to_rgb(*col)
            hexcol = rgb2hex(col)
            for c in cs:
                if len(c):
                    cc = [((x[1] - 512) / 1024 * ex * 2, (x[0] - 512) / 1024 * ex * 2) for x in c]
                    if la.norm(c[-1] - c[0]) < 0.01:
                        self.canvas.create_polygon(cc, fill="", outline=hexcol, width=7, tag="contour")
                    else:
                        self.canvas.create_line(cc, fill=hexcol, width=7, tag="contour")
        try:
            self.canvas.tag_lower("contour")
        except tk.TclError:
            pass

        sys.stdout.flush()


    def onDragStart(self, ev):
        self.canvas.focus_set()
        self.canvas.scan_mark(ev.x, ev.y)
        self.dragging = True

    def onDragEnd(self, ev):
        self.dragging = False
    def onDragToggle(self, ev):
        if self.dragging:
            self.dragging = False
        else:
            self.canvas.focus_set()
            self.canvas.scan_mark(self.canvas.winfo_pointerx(), self.canvas.winfo_pointery())
            self.dragging = True

    def onDrag(self, ev):
        # print("Motion",ev.x,ev.y,ev.state)
        sys.stdout.flush()
        if self.dragging:
            self.canvas.scan_dragto(ev.x, ev.y, 1)
            self.adjustImg()

    def onWheel(self, ev):
        print("Wheel", ev.delta, ev.state, ev.num)
        sys.stdout.flush()
        cx, cy = self.canvas.canvasxy(ev.x, ev.y)

        sf = 1.1
        if not (ev.delta == 0):  # Wheel event
            if ev.delta < 0: sf = 1 / sf
        else:
            if ev.num == 5: sf = 1 / sf

        # scale all objects on canvas
        self.canvas.zoom(cx, cy, sf)

        self.adjustImg()
        sys.stdout.flush()

    def onConfigure(self, ev):
        self.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)
        self.canvas.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)
        self.adjustImg()

    def discardImg(self):
        self.img = None
        self.simg = None
        self.pimg = None
        self.canvas.delete(self.imgcid)
        self.imgcid = None
        self.imgbbox = None
    def importImg(self):
        path = self.askOpenFileName()

        try:
            self.img = Image.open(path)
        except FileNotFoundError:
            print("file not found!")
            return

        self.simg = self.img
        self.pimg = ImageTk.PhotoImage(self.img)
        self.imgcid = self.canvas.create_image(0, 0, image=self.pimg, anchor=tk.NW, tag="image")
        self.imgbbox = (0, 0) + self.img.size
        self.canvas.tag_lower("image", "segment")
        self.adjustImg()
    def adjustImg(self):
        if self.img:
            xf = self.canvas.xform()
            xf = deepcopy(xf)
            xf = la.inverse(xf)

            # canvas coordinate of image origin
            ox, oy = self.canvas.canvasxy(self.imgbbox[0], self.imgbbox[1])
            fx, fy = self.canvas.canvasxy(self.imgbbox[2], self.imgbbox[3])
            cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()  # max image width,height
            bx, by = self.canvas.canvasxy(0, 0)
            ex, ey = self.canvas.canvasxy(cw, ch)

            cbx, cby = self.canvas.canvasx(0), self.canvas.canvasy(0)
            cex, cey = self.canvas.canvasx(cw), self.canvas.canvasy(ch)
            cox, coy = self.canvas.canvasx(self.imgbbox[0]), self.canvas.canvasy(self.imgbbox[1])
            cfx, cfy = self.canvas.canvasx(self.imgbbox[2]), self.canvas.canvasy(self.imgbbox[3])

            ix = bx
            iy = by
            iw = cw
            ih = ch

            # print((ox,oy),(fx,fy),(bx,by),(ex,ey))
            # print((ix,iy),(iw,ih))

            # scale image contents, max size of cw,ch make sure to not overblow image size
            # self.simg = self.img.transform((iw,ih),Image.AFFINE,data=(xf[0][0],xf[0][1],xf[0][3],xf[1][0],xf[1][1],xf[1][3]))
            self.simg = self.img.transform((iw, ih), Image.AFFINE,
                                           data=(xf[0][0], xf[0][1], ix, xf[1][0], xf[1][1], iy))
            self.canvas.coords(self.imgcid, ix, iy)  # adjust image origin

            self.pimg = ImageTk.PhotoImage(self.simg)
            self.canvas.itemconfig(self.imgcid, image=self.pimg)  # set new image
            self.canvas.tag_lower(self.imgcid, "segment")  # just below segments
            # sys.stdout.flush()

    def drawCoordGrid(self):

        self.canvas.create_line(-self.extent, 0, self.extent, 0, fill="grey", tag="grid")
        self.canvas.create_line(0, -self.extent, 0, self.extent, fill="grey", tag="grid")
        for i in range(1, int(self.extent / 100)):
            self.canvas.create_line(-self.extent, i * 100, self.extent, i * 100, fill="lightgrey", tag="grid")
            self.canvas.create_line(i * 100, -self.extent, i * 100, self.extent, fill="lightgrey", tag="grid")
            self.canvas.create_line(-self.extent, -i * 100, self.extent, -i * 100, fill="lightgrey", tag="grid")
            self.canvas.create_line(-i * 100, -self.extent, -i * 100, self.extent, fill="lightgrey", tag="grid")
        self.canvas.tag_lower("grid")

root = tk.Tk()
app = App(root)

app.master.title("The TED Editor")
# app.master.maxsize(1900,1000)

import gc
# gc.set_debug(gc.DEBUG_LEAK)
gc.set_debug(gc.DEBUG_UNCOLLECTABLE)

app.mainloop()
