import math as m

from src import linalg as la

def lerp(t, a, b):
    return (1 - t) * a + t * b
def bezier2(t, ps):
    """ evaluate quadratic bezier curve at t given three points in list ps """
    umt = 1 - t
    return umt * umt * ps[0] + 2 * umt * t * ps[1] + t * t * ps[2]
def bezier2_d(t, ps):
    """ evaluate derivative of quadratic bezier curve at t given
  coordinates of three points in list ps """
    umt = 1 - t
    return 2 * umt * (ps[1] - ps[0]) + 2 * t * (ps[2] - ps[1])
def bezier2_dr(t, ps):
    """ evaluate derivative of rational quadratic bezier curve at t given
  homogeneous coordinates of three points in list ps """
    # rational curve, use quotient rule to get derivative
    cut = [la.cut(ps[0]), la.cut(ps[1]), la.cut(ps[2])]
    xt = bezier2(t, cut)
    dxt = bezier2_d(t, cut)
    w = [ps[0][-1], ps[1][-1], ps[2][-1]]  # weights
    wt = bezier2(t, w)
    dwt = bezier2_d(t, w)
    return (dxt * wt - dwt * xt) / (wt * wt)
def bezier2_h(t, hs):
    """ evaluate quadratic bezier curve at t and project result """
    x = bezier2(t, hs)
    return la.proj(x)
def circle2_h(t, hs):
    """ evaluate quadratic bezier circle at t from -1 to 1, if t < 0 use the opposite arc"""
    x = bezier2(abs(t), [hs[0], hs[1] if t > 0 else -hs[1], hs[2]])
    return la.proj(x)
def bezier2_tangent_h(t, hs):
    """ return unit vector in the direction of the tangent of rational quadratic bezier curve at t """
    dx = bezier2_dr(t, hs)
    return la.unit(dx)
def bezier_circle_from_two_points_and_tangent(p0, p1, t0):
    """ return homogeneous control points of rational quadratic bezier
  circle going through p0 and p1 with tangent t0 at p0 """
    chord = p1 - p0
    ct, cl = la.unit_length(chord)

    cosa2 = la.dot(t0, ct)  # cosine of half the angle between tangents at p0 and p1

    # gotta love homogeneous coordinates, normal calculation of midpoint looks like this:
    # pm = p0 + cl/(2*cosa2) * t
    # notice the division through cosa2 and the special cases needed when it goes to zero...
    ph = p0 * cosa2 + cl / 2 * t0
    return [la.hom(p0, 1), la.coords(ph[0], ph[1], cosa2), la.hom(p1, 1)]
def circle_center_from_two_points_and_tangent(p0, p1, t0):
    chord = p1 - p0
    ct, cl = la.unit_length(chord)
    n = la.perp2ccw(t0)
    sina2 = la.dot(n, ct)  # sine of half the angle between tangents at p0 and p1
    ch = p0 * sina2 + cl / 2 * n
    return la.coords(ch[0], ch[1], sina2)
def switch_circle_segment(hs):
    """given a rational quadratic bezier circle, switch the segment that is
  evaluated for t from 0 to 1. this enables evaluation of the whole circle"""
    hs[1] = -hs[1]
def biarc_joint_circle_tangent(p0, t0, p1, t1):
    """find the tangent of the joint circle of the biarc at p0"""
    V = p1 - p0
    eps = 1 / 2 ** 12

    pt1 = la.refl(t1, V)  # reflect t1 about chord
    d0 = t0 + pt1
    if la.norm2(d0) < eps:
        # if la.dot(t0,t1) > 0: # joint circle is a line
        #  d0 = t0
        # elif la.dot(t0,t1) < 0: # joint circle is minimal
        d0 = la.perp2ccw(t0)

    tJ0 = la.unit(d0)
    return tJ0
def biarc_joint_circle(p0, t0, p1, t1):
    """returns the circle of all biarc joint points"""
    tJ0 = biarc_joint_circle_tangent(p0, t0, p1, t1)
    return bezier_circle_from_two_points_and_tangent(p0, p1, tJ0)  # joint circle
def biarc_joint_point(Jb, r):
    """returns the joint point of a biarc given the joint circle and a
  parameter between -1 and 1"""
    return circle2_h(r, Jb)
def biarc_tangent_at_joint_point(J, p0, t0):
    """returns the tangent of both joining biarc circle segments at joint
  point J, given start point p0 and start tangent t0"""
    return la.refl(t0, J - p0)
def biarc_h(p0, t0, p1, t1, r):
    """construct biarc from p0 with tangent t0 to p1 with tangent t1, at
   joint circle rational bezier parameter r. returns both circles and
   the joint circle"""
    Jb = biarc_joint_circle(p0, t0, p1, t1)
    J = biarc_joint_point(Jb, r)
    Jt = biarc_tangent_at_joint_point(J, p0, t0)
    C1 = bezier_circle_from_two_points_and_tangent(p0, J, t0)
    C2 = bezier_circle_from_two_points_and_tangent(J, p1, Jt)
    return C1, C2, Jb
def circleparam(p0, p1, t0):
    """find circle parameters for circle given by a chord from
   p0 to p1 and tangent t0 at p0. returns center, signed curvature k,
   segment angle subtended, and the arclength

  """
    chord = p1 - p0
    ut0 = la.unit(t0)
    n = la.perp2ccw(ut0)
    chordsinha = la.dot(chord, n)
    chordcosha = la.dot(chord, ut0)
    chord2 = la.norm2(chord)
    signed_curvature = 2 * chordsinha / chord2
    lc = m.sqrt(chord2)

    ha = 0
    length = 0

    eps = 2 ** -16
    if abs(signed_curvature) > eps:
        center = p0 + n * (1 / signed_curvature)
        ha = m.acos(chordcosha / lc)
        length = 2 * ha / abs(signed_curvature)
    else:
        center = p0
        length = lc
        signed_curvature = 0

    return center, signed_curvature, 2 * ha, length
def bc_arclen_t(s, a):
    """return bezier parameter of rational bezier circle for arclength s
  and total segment angle a (angle between first and last bezier control point)"""
    if a > 0:
        ha = a / 2
        return m.sin(ha * s) / (m.sin(ha * s) + m.sin(ha * (1 - s)))
    else:
        return s
def bezier_circle_parameter(p0, p1, t0, pc):
    """given point on circle pc and bezier start and end points p0 and p1
  and tangent t0 at p0 return bezier parameter of p"""
    t = la.norm(pc - p0) / (la.norm(pc - p0) + la.norm(pc - p1))
    # first dot determines if pc is on short (>0) or long segment,
    # second determines if joint circle t > 0 is on short segment (>0) or long
    chord = p0 - p1
    f1 = -la.dot(pc - p0, pc - p1)
    f2 = -la.dot(t0, chord)
    si = f1 * f2
    eps = 2 ** -16

    if abs(si) < eps:
        # chord is 2*radius, check vector perpendicular to chord and joint circle
        # tangent at p0
        tp = la.perp(pc, chord)
        si = la.dot(tp, t0)

    if (si < 0):
        t = -t
    return t
def point_on_circle(p0, p1, t0, p):
    """finds point on circle nearest to p, returns point on circle,
  corresponding parameter value t and segment angle"""
    ch = circle_center_from_two_points_and_tangent(p0, p1, t0)

    eps = 2 ** -16

    if abs(ch[2]) < eps:  # circle is line
        top = p - p0
        chord = p1 - p0
        pc = la.para(top, chord)
        t = la.dot(pc, chord) / la.norm2(chord)
        return p0 + pc, t, 0
    else:
        c = la.proj(ch)
        tp = la.unit(p - c)
        r = la.norm(c - p0)
        pc = c + r * tp

        t = bezier_circle_parameter(p0, p1, t0, pc)
        return pc, t, la.dot(la.unit(t0), la.unit(pc - p0))
def offset_circle(hs, o):
    t0 = la.unit(bezier2_dr(0, hs))  # tangent at p0
    t1 = la.unit(bezier2_dr(0.5, hs))  # tangent at p1
    t2 = la.unit(bezier2_dr(1, hs))  # tangent at p2
    pt0 = la.hom(la.perp2ccw(t0), 0)  # offset direction at p0
    pt1 = la.hom(la.perp2ccw(t1), 0)  # offset direction at p1
    pt2 = la.hom(la.perp2ccw(t2), 0)  # offset direction at p2
    return [hs[0] + o * pt0, hs[1] + o * pt1, hs[2] + o * pt2]
def transform_bezier_circle(hs, xform):
    for i, h in enumerate(hs):
        th = la.vapply(xform, la.coords(h[0], h[1], 0, h[2]))
        hs[i] = la.coords(th[0], th[1], th[3])
    return hs
def transform_point(p, xform):
    tp = la.vapply(xform, la.coords(p[0], p[1], 0, 1))
    return la.coords(tp[0], tp[1])
def transform_vector(v, xform):
    tv = la.vapply(xform, la.coords(v[0], v[1], 0, 0))
    return la.coords(tv[0], tv[1])