"""Thin-lens optics calculator for the scanner's imaging path.

o = distance from the lens to the tooth surface measured ALONG the folded optical path
    (camera-to-mirror + mirror-to-tooth).

Why not FOV = distance * sensor / f?  That is the far-field shortcut (it assumes the image
sits at one focal length behind the lens). At close range the lens must extend, and the exact
field is sensor * (o - f) / f.

Depth resolution is the key triangulation relation:
    depth per pixel = (lateral size of one pixel at the tooth) / tan(triangulation angle)
where the triangulation angle is the angle between the laser plane and the viewing axis.

Examples:
    python scripts/optics_calc.py                                  # built-in comparison
    python scripts/optics_calc.py --f 25 --o 125 --fnum 8 --tri-angle 25
"""
import argparse
import math


def analyse(f, o, pitch_um=3.1, w_px=1600, h_px=1200, fnum=2.8, tri_deg=30.0,
            depth_pm_mm=4.0, coc_px=2.0):
    if o <= 1.05 * f:
        raise ValueError("o must be clearly larger than f")
    p = pitch_um / 1000.0
    i = o * f / (o - f)                 # lens-to-sensor distance when focused at o
    m = i / o                           # magnification
    foot_um = pitch_um / m              # lateral size of one pixel at the tooth
    c = coc_px * p                      # acceptable blur circle
    hyper = f * f / (fnum * c) + f
    near = o * (hyper - f) / (hyper + o - 2 * f)
    far = o * (hyper - f) / (hyper - o) if o < hyper else math.inf
    aperture = f / fnum

    def blur_px(delta):
        o2 = o + delta
        i2 = o2 * f / (o2 - f)
        return aperture * abs(i - i2) / i2 / p

    return {
        "fov_w_mm": w_px * p / m,
        "fov_h_mm": h_px * p / m,
        "um_per_px": foot_um,
        "fx_eff_px": i / p,
        "extension_mm": i - f,
        "dof_total_mm": far - near,
        "blur_px_at_depth": max(blur_px(+depth_pm_mm), blur_px(-depth_pm_mm)),
        "depth_um_per_px": foot_um / math.tan(math.radians(tri_deg)),
        "tri_deg": tri_deg,
    }


def show(name, r, depth_pm_mm):
    print(f"{name:<34} FOV {r['fov_w_mm']:5.1f} x {r['fov_h_mm']:4.1f} mm | {r['um_per_px']:5.1f} um/px | "
          f"fx {r['fx_eff_px']:6.0f} px | lens ext {r['extension_mm']:4.1f} mm | DOF {r['dof_total_mm']:5.2f} mm | "
          f"blur at +-{depth_pm_mm:g} mm {r['blur_px_at_depth']:5.1f} px | "
          f"tri {r['tri_deg']:4.1f} deg -> depth {r['depth_um_per_px']:6.1f} um/px "
          f"({0.1 * r['depth_um_per_px']:.1f} um per 0.1 px)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--f", type=float, help="focal length (mm)")
    ap.add_argument("--o", type=float, help="lens-to-tooth path length (mm)")
    ap.add_argument("--fnum", type=float, default=2.8)
    ap.add_argument("--tri-angle", type=float, default=30.0, help="laser plane vs viewing axis (deg)")
    ap.add_argument("--pitch", type=float, default=3.1, help="pixel pitch (um)")
    ap.add_argument("--depth-pm", type=float, default=4.0, help="+- depth variation of the teeth (mm)")
    a = ap.parse_args()

    if a.f and a.o:
        show("custom", analyse(a.f, a.o, a.pitch, fnum=a.fnum, tri_deg=a.tri_angle, depth_pm_mm=a.depth_pm), a.depth_pm)
        return
    d = a.depth_pm
    rows = [
        ("A  8mm  o=20  f/2.0  tri 31", dict(f=8, o=20, fnum=2.0, tri_deg=31)),
        ("B0 25mm o=100 f/2.8  tri 5", dict(f=25, o=100, fnum=2.8, tri_deg=5)),
        ("B1 25mm o=125 f/2.8  tri 5", dict(f=25, o=125, fnum=2.8, tri_deg=5)),
        ("B2 25mm o=125 f/8    tri 5", dict(f=25, o=125, fnum=8.0, tri_deg=5)),
        ("B3 25mm o=125 f/8    tri 25", dict(f=25, o=125, fnum=8.0, tri_deg=25)),
    ]
    for name, kw in rows:
        show(name, analyse(pitch_um=a.pitch, depth_pm_mm=d, **kw), d)


if __name__ == "__main__":
    main()