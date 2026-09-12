#!/usr/bin/env python3
"""Compose a Vega + dual SharpaWave URDF for visualisation.

MagicSim is supposed to ship `Assets/Robots/vega_1p_sharpa.usd`, but that file
is not in this checkout (noted missing 2026-07-19 and still absent). The pieces
are all here though - the bare arm URDF, and the hands as standalone URDFs with
a flange root meant for exactly this - so build the combined model instead of
waiting for the asset.

  vega_1.urdf                        arm + torso + head, EE links L_ee / R_ee
  {left,right}_sharpa_wave_with_flange.urdf   hand rooted at *_hand_flange

Joining them is a fixed joint per side from the arm's EE to the hand's flange.
`package://<pkg>/meshes/*.STL` references are rewritten to absolute paths so
any loader resolves them without a ROS package path.

VISUALISATION ONLY. The IK deliberately keeps running on the bare arm URDF:
PinkVegaIK builds its reduced model by locking a hardcoded list of non-arm
joints, and 40 unlisted finger joints would silently stay free and change the
solve. So solve on the arm, draw with the hands - the arm joint names are
identical in both, so the solution transfers unchanged.

Usage:
    .venv/bin/python scripts/build_vega_sharpa_urdf.py
    .venv/bin/python scripts/build_vega_sharpa_urdf.py --out /tmp/x.urdf
"""
from __future__ import annotations

import argparse
import pathlib
import xml.etree.ElementTree as ET

VEGA = pathlib.Path(
    "~/Dexmate/dexmate-urdf/robots/humanoid/vega_1/vega_1.urdf").expanduser()
SHARPA_ROOT = pathlib.Path("~/dexmate/sharpa-urdf-usd-xml/wave_01").expanduser()
HANDS = {
    "right": ("right_sharpa_wave/right_sharpa_wave_with_flange.urdf",
              "right_hand_flange", "R_ee"),
    "left": ("left_sharpa_wave/left_sharpa_wave_with_flange.urdf",
             "left_hand_flange", "L_ee"),
}
DEFAULT_OUT = pathlib.Path(__file__).resolve().parents[1] / "assets/vega_1_sharpa.urdf"


def absolutise_meshes(root: ET.Element, urdf_dir: pathlib.Path) -> int:
    """Make every mesh path absolute.

    Two forms appear: `package://<pkg>/rest` (the Sharpa hands, a ROS package)
    and plain `meshes/x.obj` relative to the URDF (Vega). Both resolve only
    from their original directory, and the combined file lives somewhere else,
    so both have to be rewritten or the viewer silently draws nothing.
    """
    n = 0
    for mesh in root.iter("mesh"):
        fn = mesh.get("filename", "")
        if fn.startswith("package://"):
            rest = fn[len("package://"):].split("/", 1)[1]
        elif fn and not fn.startswith(("/", "file://")):
            rest = fn
        else:
            continue
        mesh.set("filename", str((urdf_dir / rest).resolve()))
        n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vega", type=pathlib.Path, default=VEGA)
    ap.add_argument("--out", type=pathlib.Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    if not args.vega.exists():
        raise SystemExit(f"missing {args.vega}")
    tree = ET.parse(args.vega)
    root = tree.getroot()
    root.set("name", "vega_1_sharpa")
    n = absolutise_meshes(root, args.vega.parent)
    have = {ln.get("name") for ln in root.findall("link")}
    print(f"[build] vega: {len(have)} links, {len(root.findall('joint'))} joints, "
          f"{n} mesh paths made absolute")

    for side, (rel, flange, ee) in HANDS.items():
        path = SHARPA_ROOT / rel
        if not path.exists():
            raise SystemExit(f"missing {path}")
        if ee not in have:
            raise SystemExit(f"{ee} not in the arm URDF - cannot attach {side} hand")
        hroot = ET.parse(path).getroot()
        nh = absolutise_meshes(hroot, path.parent)
        added_l = added_j = 0
        for el in list(hroot):
            if el.tag == "link":
                if el.get("name") in have:
                    raise SystemExit(
                        f"link name collision joining {side} hand: {el.get('name')}")
                have.add(el.get("name"))
                root.append(el)
                added_l += 1
            elif el.tag == "joint":
                root.append(el)
                added_j += 1
            # <mujoco>, <material> etc. are dropped: loader-specific, and the
            # arm URDF already carries what a viewer needs
        weld = ET.SubElement(root, "joint")
        weld.set("name", f"{side}_sharpa_mount")
        weld.set("type", "fixed")
        ET.SubElement(weld, "parent").set("link", ee)
        ET.SubElement(weld, "child").set("link", flange)
        # EE frame and flange frame are both at the mounting face, so identity.
        # If the hands end up visibly rotated, this origin is the knob to turn.
        o = ET.SubElement(weld, "origin")
        o.set("xyz", "0 0 0")
        o.set("rpy", "0 0 0")
        print(f"[build] {side}: +{added_l} links +{added_j} joints, "
              f"{nh} meshes, welded {ee} -> {flange}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(args.out, encoding="utf-8", xml_declaration=True)
    print(f"\n[build] wrote {args.out}")
    print(f"[build] total {len(root.findall('link'))} links, "
          f"{len(root.findall('joint'))} joints")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
