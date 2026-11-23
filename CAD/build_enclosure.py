"""
QTennis wrist pod - parametric FreeCAD build script.

Run inside FreeCAD (GUI, because vendor STEPs are imported with ImportGui):
    exec(open("/home/neckles/projects/qtennis/CAD/build_enclosure.py").read())

Everything is driven by the `Params` spreadsheet. Every enclosure feature is a
PartDesign Pad/Pocket/Hole on a fully constrained Sketch whose dimensions are
expressions bound to Params, so edit a cell -> recompute -> the model follows.

Frame (cavity frame):
    origin = inner cavity corner on the floor's -X/-Y side, z=0 at the skin side
    X  across the pod (along the forearm), Y = strap direction, Z = up from wrist.

Components (BOM "Shelton setup"):
    5426 QT Py ESP32-S3          official Adafruit STEP (Adafruit_CAD_Parts)
    5397 LiPo Charger BFF        official Adafruit STEP, mounted under the QT Py
    1578 500mAh LiPo             official Adafruit STEP
    4554 ICM-20948 STEMMA QT     sketch proxy from Adafruit-ICM20948-PCB .brd
    2305 DRV2605L STEMMA QT      sketch proxy from Adafruit-DRV2605-PCB .brd
    1201 vibrating motor disc    sketch proxy (10 mm x 2.7 mm)
"""
import os
import FreeCAD as App
import Part
import Sketcher

V = App.Vector
CAD = "/home/neckles/projects/qtennis/CAD"
VEND = os.path.join(CAD, "vendor")
DOCNAME = "QTennis_Enclosure"

# --------------------------------------------------------------------------
# Parameters: (alias, value or "=formula", note)
# --------------------------------------------------------------------------
PARAMS = [
    ("# shell", None, ""),
    ("wall", 2.0, "side wall thickness"),
    ("floor", 1.5, "floor thickness (skin side)"),
    ("lid_t", 1.5, "lid plate thickness"),
    ("gap", 0.4, "PCB / battery to wall clearance"),
    ("r_out", 3.0, "outer corner radius"),
    ("r_in", 1.0, "cavity corner radius"),
    ("# PCB generics", None, ""),
    ("pcb_t", 1.57, "PCB thickness"),
    ("sh_h", 2.96, "JST-SH (STEMMA QT) receptacle height above PCB"),
    ("brd_L", 25.4, "Adafruit 1.0in STEMMA board length (IMU, DRV)"),
    ("brd_W", 17.78, "Adafruit 0.7in STEMMA board width"),
    ("brd_r", 2.54, "board corner radius"),
    ("brd_hole", 2.5, "board mounting hole dia"),
    ("brd_inset", 2.54, "mounting hole inset from board edge"),
    ("# QT Py + LiPo BFF stack", None, ""),
    ("qt_w", 17.78, "QT Py / BFF width"),
    ("qt_len", 20.70, "QT Py / BFF length"),
    ("hdr_gap", 2.5, "header plastic spacer between QT Py and BFF"),
    ("bff_comp", 4.80, "tallest BFF part (JST-PH) - BFF parts face the floor"),
    ("sh_clr", 5.0, "-Y keep-out: QT Py STEMMA plug + cable bend"),
    ("ph_clr", 4.3, "+Y keep-out: battery JST-PH plug under the USB-C"),
    ("zone_gap", 2.0, "wire channel between stack and battery zone"),
    ("# battery zone", None, ""),
    ("batt_L", 36.0, "1578 LiPo length"),
    ("batt_W", 29.0, "1578 LiPo width"),
    ("batt_H", 4.75, "1578 LiPo thickness"),
    ("batt_gap", 0.3, "air gap above IMU connectors / below DRV parts"),
    ("chan", 3.4, "+X riser channel for DRV->IMU STEMMA cable"),
    ("imu_boss", 0.8, "IMU standoff height above floor"),
    ("imu_plug", 6.0, "+X keep-out for the IMU STEMMA plug"),
    ("drv_boss", 0.6, "DRV standoff below lid"),
    ("mot_d", 10.0, "1201 motor disc dia"),
    ("mot_h", 2.7, "1201 motor disc thickness"),
    ("mot_recess", 0.4, "motor pocket depth into floor"),
    ("post_w", 3.0, "battery support post size"),
    ("# strap (1in nylon strap)", None, ""),
    ("strap_w", 25.4, "strap width"),
    ("slot_t", 3.4, "strap slot thickness"),
    ("bar_t", 2.6, "lug bar thickness"),
    ("lug_h", 5.0, "lug height"),
    ("lug_side", 3.0, "lug cheek width"),
    ("# fasteners (M2)", None, ""),
    ("lobe_off", 2.4, "screw axis offset from cavity corner (diagonal)"),
    ("lobe_r", 3.0, "corner lobe radius"),
    ("pilot_d", 1.7, "M2 self-tap pilot in base"),
    ("clear_d", 2.3, "M2 clearance"),
    ("csk_d", 4.2, "M2 countersink dia"),
    ("# openings", None, ""),
    ("usb_w", 13.0, "USB-C cable overmold opening width"),
    ("usb_h", 7.5, "USB-C opening height (notch, open to lid)"),
    ("sw_w", 6.0, "slide switch access slot width"),
    ("sw_h", 3.4, "slide switch access slot height"),
    ("btn_d", 2.0, "QT Py BOOT/RESET pin hole"),
    ("# derived", None, ""),
    ("Xc", "=gap + qt_w + zone_gap + gap + batt_L + chan", "cavity X"),
    ("Yc", "=sh_clr + qt_len + ph_clr", "cavity Y"),
    ("h_in", "=imu_boss + pcb_t + sh_h + batt_gap + batt_H + batt_gap + sh_h + pcb_t + drv_boss", "cavity height"),
    ("base_h", "=floor + h_in", "base height (lid seat)"),
    ("total_h", "=base_h + lid_t", "overall height"),
    ("bff_z0", "=floor + bff_comp", "BFF component face"),
    ("qt_z0", "=floor + bff_comp + pcb_t + hdr_gap", "QT Py board underside"),
    ("rz_x0", "=gap + qt_w + zone_gap", "battery zone start"),
    ("batt_x0", "=rz_x0 + gap", ""),
    ("batt_y0", "=(Yc - batt_W) / 2", ""),
    ("batt_z0", "=floor + imu_boss + pcb_t + sh_h + batt_gap", ""),
    ("imu_x0", "=Xc - imu_plug - brd_L", ""),
    ("imu_y0", "=gap", ""),
    ("imu_z0", "=floor + imu_boss", ""),
    ("drv_x0", "=batt_x0 + (batt_L - brd_L) / 2", ""),
    ("drv_y0", "=batt_y0 + (batt_W - brd_W) / 2", ""),
    ("drv_z0", "=base_h - drv_boss", "DRV back face (hangs from lid)"),
    ("mot_x", "=imu_x0 + brd_L / 2", "motor centre"),
    ("mot_y", "=Yc - gap - mot_d / 2 - 0.3", "motor centre"),
    ("slot_w", "=strap_w + 1.6", ""),
    ("lug_wx", "=slot_w + 2 * lug_side", ""),
    ("qt_cx", "=gap + qt_w / 2", "stack centreline X (USB, switch)"),
]


def E(s):
    """'$alias' shorthand -> 'Params.alias'."""
    return str(s).replace("$", "Params.")


class SK:
    """Sketch builder: every dimension is an expression bound to Params."""

    def __init__(self, body, name, plane="XY", offset=None):
        self.body = body
        self.ss = body.Document.getObject("Params")
        planes = {f.Role: f for f in body.Origin.OriginFeatures}
        self.sk = body.newObject("Sketcher::SketchObject", name)
        self.sk.AttachmentSupport = [(planes[plane + "_Plane"], "")]
        self.sk.MapMode = "FlatFace"
        if offset is not None:
            self.sk.setExpression(".AttachmentOffset.Base.z", E(offset))
        self.n = 0

    def val(self, expr):
        q = self.ss.evalExpression(E(expr))
        return float(q.Value if hasattr(q, "Value") else q)

    def con(self, c, expr=None):
        i = self.sk.addConstraint(c)
        if expr is not None:
            self.sk.setExpression(f"Constraints[{i}]", E(expr))
        return i

    def rect(self, x0, y0, w, h):
        X0, Y0, W, H = (self.val(e) for e in (x0, y0, w, h))
        p = [V(X0, Y0, 0), V(X0 + W, Y0, 0), V(X0 + W, Y0 + H, 0), V(X0, Y0 + H, 0)]
        g = [self.sk.addGeometry(Part.LineSegment(p[i], p[(i + 1) % 4])) for i in range(4)]
        for i in range(4):
            self.con(Sketcher.Constraint("Coincident", g[i], 2, g[(i + 1) % 4], 1))
        self.con(Sketcher.Constraint("Horizontal", g[0]))
        self.con(Sketcher.Constraint("Horizontal", g[2]))
        self.con(Sketcher.Constraint("Vertical", g[1]))
        self.con(Sketcher.Constraint("Vertical", g[3]))
        self.con(Sketcher.Constraint("DistanceX", g[0], 1, g[0], 2, W), w)
        self.con(Sketcher.Constraint("DistanceY", g[1], 1, g[1], 2, H), h)
        self.con(Sketcher.Constraint("DistanceX", -1, 1, g[0], 1, X0), x0)
        self.con(Sketcher.Constraint("DistanceY", -1, 1, g[0], 1, Y0), y0)
        return g

    def rrect(self, x0, y0, w, h, r):
        """Rounded rectangle, lower-left corner (x0, y0)."""
        X0, Y0, W, H, R = (self.val(e) for e in (x0, y0, w, h, r))
        import math
        L = [
            Part.LineSegment(V(X0 + R, Y0, 0), V(X0 + W - R, Y0, 0)),          # bottom
            Part.LineSegment(V(X0 + W, Y0 + R, 0), V(X0 + W, Y0 + H - R, 0)),  # right
            Part.LineSegment(V(X0 + W - R, Y0 + H, 0), V(X0 + R, Y0 + H, 0)),  # top
            Part.LineSegment(V(X0, Y0 + H - R, 0), V(X0, Y0 + R, 0)),          # left
        ]
        cs = [V(X0 + W - R, Y0 + R, 0), V(X0 + W - R, Y0 + H - R, 0),
              V(X0 + R, Y0 + H - R, 0), V(X0 + R, Y0 + R, 0)]
        a0s = [-90, 0, 90, 180]
        A = [Part.ArcOfCircle(Part.Circle(c, V(0, 0, 1), R), math.radians(a), math.radians(a + 90))
             for c, a in zip(cs, a0s)]
        gl = [self.sk.addGeometry(l) for l in L]
        ga = [self.sk.addGeometry(a) for a in A]   # BR, TR, TL, BL
        # chain: bottom -> BR -> right -> TR -> top -> TL -> left -> BL -> bottom
        for i in range(4):
            self.con(Sketcher.Constraint("Tangent", gl[i], 2, ga[i], 1))
            self.con(Sketcher.Constraint("Tangent", ga[i], 2, gl[(i + 1) % 4], 1))
        self.con(Sketcher.Constraint("Horizontal", gl[0]))
        self.con(Sketcher.Constraint("Horizontal", gl[2]))
        self.con(Sketcher.Constraint("Vertical", gl[1]))
        self.con(Sketcher.Constraint("Vertical", gl[3]))
        for k in (1, 2, 3):
            self.con(Sketcher.Constraint("Equal", ga[0], ga[k]))
        self.con(Sketcher.Constraint("Radius", ga[0], R), r)
        self.con(Sketcher.Constraint("DistanceX", ga[3], 3, ga[0], 3, W - 2 * R), f"({w}) - 2*({r})")
        self.con(Sketcher.Constraint("DistanceY", ga[0], 3, ga[1], 3, H - 2 * R), f"({h}) - 2*({r})")
        self.con(Sketcher.Constraint("DistanceX", -1, 1, ga[3], 3, X0 + R), f"({x0}) + ({r})")
        self.con(Sketcher.Constraint("DistanceY", -1, 1, ga[3], 3, Y0 + R), f"({y0}) + ({r})")
        return gl + ga

    def circle(self, cx, cy, d):
        CX, CY, D = (self.val(e) for e in (cx, cy, d))
        g = self.sk.addGeometry(Part.Circle(V(CX, CY, 0), V(0, 0, 1), D / 2))
        self.con(Sketcher.Constraint("Diameter", g, D), d)
        self.con(Sketcher.Constraint("DistanceX", -1, 1, g, 3, CX), cx)
        self.con(Sketcher.Constraint("DistanceY", -1, 1, g, 3, CY), cy)
        return g

    def done(self):
        self.sk.recompute()
        return self.sk


def pad(body, name, sk, length, reversed_=False):
    f = body.newObject("PartDesign::Pad", name)
    f.Profile = sk
    f.setExpression("Length", E(length))
    f.Reversed = reversed_
    f.Refine = True
    sk.Visibility = False
    body.Document.recompute()
    return f


def pocket(body, name, sk, length=None, through=False, midplane=False, reversed_=False):
    f = body.newObject("PartDesign::Pocket", name)
    f.Profile = sk
    if through:
        f.Type = "ThroughAll"
    else:
        f.setExpression("Length", E(length))
    f.Midplane = midplane
    f.Reversed = reversed_
    f.Refine = True
    sk.Visibility = False
    body.Document.recompute()
    return f


def hole(body, name, sk, dia, csk=None, reversed_=False):
    f = body.newObject("PartDesign::Hole", name)
    f.Profile = sk
    f.Threaded = False
    f.setExpression("Diameter", E(dia))
    f.DepthType = "ThroughAll"
    if csk:
        f.HoleCutType = "Countersink"
        f.setExpression("HoleCutDiameter", E(csk))
        f.HoleCutCountersinkAngle = 90
    f.Reversed = reversed_
    f.Refine = True
    sk.Visibility = False
    body.Document.recompute()
    return f


def corners(ox, oy, w, h, inset):
    """Four points inset from a rectangle's corners (expression strings)."""
    return [(f"{ox} + {inset}", f"{oy} + {inset}"),
            (f"{ox} + {w} - {inset}", f"{oy} + {inset}"),
            (f"{ox} + {inset}", f"{oy} + {h} - {inset}"),
            (f"{ox} + {w} - {inset}", f"{oy} + {h} - {inset}")]


LOBES = [("-$lobe_off", "-$lobe_off"), ("$Xc + $lobe_off", "-$lobe_off"),
         ("-$lobe_off", "$Yc + $lobe_off"), ("$Xc + $lobe_off", "$Yc + $lobe_off")]


def outline(body, prefix, z):
    """Outer rounded rectangle + corner screw lobes, padded from height z."""
    s = SK(body, prefix + "OutlineSketch", "XY", z)
    s.rrect("-$wall", "-$wall", "$Xc + 2*$wall", "$Yc + 2*$wall", "$r_out")
    o = s.done()
    s = SK(body, prefix + "LobeSketch", "XY", z)
    for cx, cy in LOBES:
        s.circle(cx, cy, "2*$lobe_r")
    l = s.done()
    return o, l


# --------------------------------------------------------------------------
def build():
    if DOCNAME in App.listDocuments():
        App.closeDocument(DOCNAME)
    doc = App.newDocument(DOCNAME)

    ss = doc.addObject("Spreadsheet::Sheet", "Params")
    ss.set("A1", "alias"); ss.set("B1", "value [mm]"); ss.set("C1", "note")
    row = 2
    for a, v, note in PARAMS:
        if v is None:
            ss.set(f"A{row}", a)
        else:
            ss.set(f"A{row}", a)
            ss.set(f"B{row}", str(v))
            ss.setAlias(f"B{row}", a)
            ss.set(f"C{row}", note)
        row += 1
    doc.recompute()

    # ================================ BASE ================================
    base = doc.addObject("PartDesign::Body", "Base")
    o, l = outline(base, "Base", None)
    pad(base, "BaseShell", o, "$base_h")
    pad(base, "BaseLobes", l, "$base_h")

    s = SK(base, "LugSketch", "XY")
    s.rect("$Xc/2 - $lug_wx/2", "-$wall - $slot_t - $bar_t", "$lug_wx", "$slot_t + $bar_t + $wall/2")
    s.rect("$Xc/2 - $lug_wx/2", "$Yc + $wall/2", "$lug_wx", "$slot_t + $bar_t + $wall/2")
    pad(base, "StrapLugs", s.done(), "$lug_h")

    s = SK(base, "CavitySketch", "XY", "$base_h")
    s.rrect("0", "0", "$Xc", "$Yc", "$r_in")
    pocket(base, "Cavity", s.done(), "$h_in")

    s = SK(base, "StrapSlotSketch", "XY")
    s.rect("$Xc/2 - $slot_w/2", "-$wall - $slot_t", "$slot_w", "$slot_t")
    s.rect("$Xc/2 - $slot_w/2", "$Yc + $wall", "$slot_w", "$slot_t")
    pocket(base, "StrapSlots", s.done(), through=True, midplane=True)

    s = SK(base, "LidScrewPilotSketch", "XY", "$base_h")
    for cx, cy in LOBES:
        s.circle(cx, cy, "$pilot_d")
    pocket(base, "LidScrewPilots", s.done(), "$base_h - 1.0")

    imu_pts = corners("$imu_x0", "$imu_y0", "$brd_L", "$brd_W", "$brd_inset")
    s = SK(base, "IMUBossSketch", "XY", "$floor")
    for cx, cy in imu_pts:
        s.circle(cx, cy, "4.6")
    pad(base, "IMUBosses", s.done(), "$imu_boss")

    s = SK(base, "BattPostSketch", "XY", "$floor")
    s.rect("$rz_x0 + 0.5", "0.6", "$post_w", "$post_w")
    s.rect("$rz_x0 + 0.5", "$Yc/2 - $post_w/2", "$post_w", "$post_w")
    s.rect("$rz_x0 + 0.5", "$Yc - $post_w - 0.6", "$post_w", "$post_w")
    s.rect("$batt_x0 + $batt_L - $post_w - 0.5", "$Yc - $post_w - 0.6", "$post_w", "$post_w")
    pad(base, "BatteryPosts", s.done(), "$batt_z0 - $floor")

    # rails under the BFF board's long edges (BFF parts hang below its board)
    s = SK(base, "StackRailSketch", "XY", "$floor")
    s.rect("$gap", "$sh_clr + 2.5", "1.2", "$qt_len - 5")
    s.rect("$gap + $qt_w - 1.2", "$sh_clr + 2.5", "1.2", "$qt_len - 5")
    pad(base, "StackRails", s.done(), "$bff_comp")

    s = SK(base, "MotorPocketSketch", "XY", "$floor")
    s.circle("$mot_x", "$mot_y", "$mot_d + 0.6")
    pocket(base, "MotorPocket", s.done(), "$mot_recess")

    # IMU screws: M2 countersunk from the skin side, nut on top of the IMU
    s = SK(base, "IMUScrewSketch", "XY")
    for cx, cy in imu_pts:
        s.circle(cx, cy, "$clear_d")
    hole(base, "IMUScrewHoles", s.done(), "$clear_d", csk="$csk_d", reversed_=True)

    # USB-C notch through the +Y wall (open to the lid) - XZ sketch, normal -Y
    s = SK(base, "USBSketch", "XZ", "-($Yc + $wall/2)")
    s.rect("$qt_cx - $usb_w/2", "$qt_z0 + 2.67 - $usb_h/2", "$usb_w", "$base_h + 1 - ($qt_z0 + 2.67 - $usb_h/2)")
    pocket(base, "USBOpening", s.done(), "$wall + 2", midplane=True)

    s = SK(base, "SwitchSketch", "XZ", "$wall/2")
    s.rect("$qt_cx - $sw_w/2", "$floor + 3.7 - $sw_h/2", "$sw_w", "$sw_h")
    pocket(base, "SwitchSlot", s.done(), "$wall + 2", midplane=True)

    # ================================ LID =================================
    lid = doc.addObject("PartDesign::Body", "Lid")
    o, l = outline(lid, "Lid", "$base_h")
    pad(lid, "LidPlate", o, "$lid_t")
    pad(lid, "LidLobes", l, "$lid_t")

    drv_pts = corners("$drv_x0", "$drv_y0", "$brd_L", "$brd_W", "$brd_inset")
    s = SK(lid, "DRVBossSketch", "XY", "$base_h")
    for cx, cy in drv_pts:
        s.circle(cx, cy, "4.6")
    pad(lid, "DRVBosses", s.done(), "$drv_boss", reversed_=True)

    s = SK(lid, "QTPyRibSketch", "XY", "$base_h")
    s.rect("$gap + 0.5", "$sh_clr + 2", "1.2", "$qt_len - 5")
    s.rect("$gap + $qt_w - 1.7", "$sh_clr + 2", "1.2", "$qt_len - 5")
    pad(lid, "QTPyHoldDownRibs", s.done(), "$base_h - $qt_z0 - $pcb_t - 0.1", reversed_=True)

    s = SK(lid, "LidScrewSketch", "XY", "$total_h")
    for cx, cy in LOBES:
        s.circle(cx, cy, "$clear_d")
    hole(lid, "LidScrewHoles", s.done(), "$clear_d", csk="$csk_d")

    s = SK(lid, "DRVScrewSketch", "XY", "$total_h")
    for cx, cy in drv_pts:
        s.circle(cx, cy, "$clear_d")
    hole(lid, "DRVScrewHoles", s.done(), "$clear_d", csk="$csk_d")

    s = SK(lid, "ButtonHoleSketch", "XY", "$total_h")
    s.circle("$gap + 6.1", "$sh_clr + 7.11", "$btn_d")     # BOOT
    s.circle("$gap + 6.1", "$sh_clr + 11.05", "$btn_d")    # RESET
    pocket(lid, "ButtonHoles", s.done(), through=True)

    # ============================ PROXY PARTS =============================
    def stemma_board(name, conn_x, chip, color):
        b = doc.addObject("PartDesign::Body", name)
        s = SK(b, name + "PCBSketch", "XY")
        s.rrect("0", "0", "$brd_L", "$brd_W", "$brd_r")
        pad(b, name + "PCB", s.done(), "$pcb_t")
        s = SK(b, name + "HoleSketch", "XY", "$pcb_t")
        for cx, cy in corners("0", "0", "$brd_L", "$brd_W", "$brd_inset"):
            s.circle(cx, cy, "$brd_hole")
        pocket(b, name + "Holes", s.done(), through=True)
        s = SK(b, name + "PartsSketch", "XY", "$pcb_t")
        for cx in conn_x:   # JST-SH4 side-entry receptacles, 4.25 x 6.0
            s.rect(f"{cx} - 2.125", "$brd_W/2 - 3", "4.25", "6.0")
        pad(b, name + "STEMMAConnectors", s.done(), "$sh_h")
        s = SK(b, name + "ChipSketch", "XY", "$pcb_t")
        s.rect(*chip)
        pad(b, name + "Chip", s.done(), "0.9")
        b.ViewObject.ShapeColor = color
        return b

    imu = stemma_board("IMU_ICM20948", ["22.86 - 0.3", "2.54 + 0.3"],
                       ("12.573 - 1.5", "8.89 - 1.5", "3", "3"), (0.1, 0.1, 0.45))
    imu.setExpression(".Placement.Base.x", E("$imu_x0"))
    imu.setExpression(".Placement.Base.y", E("$imu_y0"))
    imu.setExpression(".Placement.Base.z", E("$imu_z0"))

    drv = stemma_board("DRV2605L_STEMMA", ["22.733 - 0.3", "2.667 + 0.3"],
                       ("7.39", "5.68", "3", "4.9"), (0.1, 0.1, 0.45))
    # hangs from the lid, components facing down: 180 deg about X
    drv.Placement = App.Placement(V(0, 0, 0), App.Rotation(V(1, 0, 0), 180))
    drv.setExpression(".Placement.Base.x", E("$drv_x0"))
    drv.setExpression(".Placement.Base.y", E("$drv_y0 + $brd_W"))
    drv.setExpression(".Placement.Base.z", E("$drv_z0"))

    mot = doc.addObject("PartDesign::Body", "Motor_1201")
    s = SK(mot, "MotorSketch", "XY")
    s.circle("0", "0", "$mot_d")
    pad(mot, "MotorDisc", s.done(), "$mot_h")
    mot.ViewObject.ShapeColor = (0.75, 0.75, 0.78)
    mot.setExpression(".Placement.Base.x", E("$mot_x"))
    mot.setExpression(".Placement.Base.y", E("$mot_y"))
    mot.setExpression(".Placement.Base.z", E("$floor - $mot_recess"))

    # ========================= VENDOR STEP PARTS ==========================
    import ImportGui

    def import_step(fname):
        before = set(o.Name for o in doc.Objects)
        ImportGui.insert(os.path.join(VEND, fname), doc.Name)
        new = [o for o in doc.Objects if o.Name not in before]
        tops = [o for o in new if not any(p.Name in [n.Name for n in new] for p in o.InList)]
        return tops[0] if len(tops) == 1 else tops

    qt = import_step("5426 QT Py ESP32-S3.step")
    qt.Label = "QTPy_ESP32S3_5426"
    qt.setExpression(".Placement.Base.x", E("$gap + 10.8585"))
    qt.setExpression(".Placement.Base.y", E("$sh_clr + 0.762"))
    qt.setExpression(".Placement.Base.z", E("$qt_z0"))

    bff = import_step("5397 Lipo Charger BFF.step")
    bff.Label = "LiPo_BFF_5397"
    bff.Placement = App.Placement(V(0, 0, 0), App.Rotation(V(0, 1, 0), 180))  # parts face floor
    bff.setExpression(".Placement.Base.x", E("$gap + $qt_w"))
    bff.setExpression(".Placement.Base.y", E("$sh_clr"))
    bff.setExpression(".Placement.Base.z", E("$bff_z0 + $pcb_t"))

    bat = import_step("1578 500mAh battery.step")
    bat.Label = "LiPo_500mAh_1578"
    bat.setExpression(".Placement.Base.x", E("$batt_x0"))
    bat.setExpression(".Placement.Base.y", E("$batt_y0"))
    bat.setExpression(".Placement.Base.z", E("$batt_z0"))

    grp = doc.addObject("App::DocumentObjectGroup", "Components")
    for o in (qt, bff, bat, imu, drv, mot):
        grp.addObject(o)

    doc.recompute()
    base.ViewObject.ShapeColor = (0.85, 0.88, 0.92)
    base.ViewObject.Transparency = 55
    lid.ViewObject.ShapeColor = (0.85, 0.88, 0.92)
    lid.ViewObject.Transparency = 70
    return doc


doc = build()
