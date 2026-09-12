#!/usr/bin/env python3
"""Azure Kinect (k4a) -> colored point cloud.

WHY THIS FILE EXISTS
    The teleop rig needs an RGBD stream that can be recorded alongside the arm
    and the hand.  The camera turned out to be an Azure Kinect DK, so none of
    the existing camera code applies - everything under V2AP-demo/camera/ is
    ZED (a different SDK, a different receiver, a different depth format).

WHICH BINDING
    pyk4a          (etiennedub)   - compiled wrapper.  DEFAULT, and the one
                                    that is confirmed to work on the rig.
                                    Point clouds come straight out of the SDK
                                    transformation API.
                                    Needs libk4a1.4-dev to build.
    pykinectazure  (ibaiGorordo)  - pure ctypes, dlopen's libk4a.so at runtime.
                                    No compile step, no -dev headers.  Use it
                                    with --binding pykinectazure if pyk4a will
                                    not build.
    Azure-Kinect-Python (hexops)  - archived, and only wraps body tracking.
                                    Not usable for point clouds.

TWO THINGS THAT SILENTLY GO WRONG HERE
    1. UNITS.  k4a hands back int16 MILLIMETRES.  This repo works in metres
       everywhere else.  Everything below is metres, and [cloud] prints the
       depth range so a factor of 1000 cannot pass unnoticed.
    2. FRAME.  These points live in the DEPTH CAMERA OPTICAL frame
       (x right, y down, z forward), not in the robot frame.  Putting the
       cloud next to the robot needs a camera->robot extrinsic that nobody has
       measured yet.  "The cloud came out" is not "the cloud is in the right
       place".

Usage:
    # 0) environment check - needs NO camera, run it first
    .venv/bin/python scripts/kinect_pointcloud.py --probe

    # 1) self-check - needs NO camera.  Synthesises a depth image whose answer
    #    is known, runs the real reprojection + PLY writer, and verifies it.
    .venv/bin/python scripts/kinect_pointcloud.py --self-check

    # 2) real camera
    .venv/bin/python scripts/kinect_pointcloud.py --out /tmp/cloud.ply
    .venv/bin/python scripts/kinect_pointcloud.py --align color --voxel 0.01
    .venv/bin/python scripts/kinect_pointcloud.py --frames 30 --out /tmp/seq.ply
"""
from __future__ import annotations

import argparse
import ctypes.util
import importlib.util
import pathlib
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))   # 为了 import magicdexmate.control_link

# ---------------------------------------------------------------------------
# nominal NFOV_UNBINNED intrinsics - USED ONLY BY --self-check.
# The real capture path never uses these: the SDK reprojects with the factory
# calibration stored in the device.  They are here so the self-check has a
# pinhole model to build a known-answer depth image with.
# ---------------------------------------------------------------------------
_MOCK_W, _MOCK_H = 640, 576
_MOCK_FX = _MOCK_FY = 504.0
_MOCK_CX, _MOCK_CY = 320.0, 288.0


@dataclass
class Cloud:
    """A colored point cloud in the depth-camera optical frame, in METRES."""

    xyz: np.ndarray  # (N, 3) float32, metres
    rgb: np.ndarray  # (N, 3) uint8
    src_shape: tuple[int, int]  # (H, W) of the image it came from
    n_raw: int  # pixels before dropping invalid depth

    def __len__(self) -> int:
        return int(self.xyz.shape[0])


# ---------------------------------------------------------------------------
# environment probe
# ---------------------------------------------------------------------------
def _find_libk4a() -> str | None:
    found = ctypes.util.find_library("k4a")
    if found:
        return found
    for cand in (
        "/usr/lib/x86_64-linux-gnu/libk4a.so",
        "/usr/lib/x86_64-linux-gnu/libk4a.so.1.4",
        "/usr/lib/aarch64-linux-gnu/libk4a.so",
    ):
        if pathlib.Path(cand).exists():
            return cand
    return None


def probe() -> int:
    """Report what is present and what is missing.  Returns an exit code."""
    print("[probe] Azure Kinect environment")
    missing: list[str] = []

    lib = _find_libk4a()
    print(f"  libk4a           : {lib or 'NOT FOUND'}"
          f"{'' if lib else '   <-- SDK NOT INSTALLED'}")
    if not lib:
        missing.append("sdk")

    udev = pathlib.Path("/etc/udev/rules.d/99-k4a.rules")
    print(f"  udev rule        : {'yes' if udev.exists() else 'NO'}"
          f"{'' if udev.exists() else '   <-- device will need root'}")
    if not udev.exists():
        missing.append("udev")

    for mod, label in (("pyk4a", "pyk4a"), ("pykinect_azure", "pykinectazure")):
        have = importlib.util.find_spec(mod) is not None
        print(f"  {label:17s}: {'importable' if have else 'NOT INSTALLED'}")
    if importlib.util.find_spec("pyk4a") is None and importlib.util.find_spec("pykinect_azure") is None:
        missing.append("binding")

    # is the camera even plugged in?  (Microsoft vendor id 045e)
    usb = "unknown (lsusb not available)"
    if shutil.which("lsusb"):
        try:
            out = subprocess.run(["lsusb"], capture_output=True, text=True, timeout=5).stdout
            hits = [ln for ln in out.splitlines() if "045e" in ln.lower() or "microsoft" in ln.lower()]
            usb = "\n                     ".join(hits) if hits else "NO Microsoft USB device  <-- CAMERA NOT PLUGGED IN"
        except Exception as exc:  # pragma: no cover - diagnostics only
            usb = f"lsusb failed: {exc}"
    print(f"  usb              : {usb}")

    if importlib.util.find_spec("pyk4a") is not None:
        try:
            import pyk4a

            n = pyk4a.connected_device_count()
            print(f"  devices seen     : {n}{'   <-- NO DEVICE' if n == 0 else ''}")
            # A probe that says "ready" with nothing plugged in is worse than no
            # probe: the software half is only half the answer, and every later
            # failure would then look like a code problem.
            if n == 0:
                missing.append("device")
        except Exception as exc:
            print(f"  devices seen     : query failed: {exc}")
            missing.append("device")

    if not missing:
        print("[probe] ready.")
        return 0
    if missing == ["device"]:
        print("\n[probe] 软件环境已就绪,但未检测到相机。\n"
              "  * 接入 Azure Kinect,并确认电源适配器也已连接。仅靠 USB 供电不足,\n"
              "    会导致设备枚举失败或连接时断时续。\n"
              "  * 若 udev 规则是本次新装的,需重新插拔一次相机才会生效。\n"
              "  * 确认接在 USB3 接口上。USB2 带宽不足,深度流会掉帧或无法启动。\n"
              "  * 可用 lsusb | grep -i microsoft 复查。")
        return 1

    print("\n[probe] NOT READY. Fix, in order:")
    if "sdk" in missing or "udev" in missing:
        print("""
  # --- Azure Kinect SDK.  Ubuntu 22.04 is not officially supported by
  # --- Microsoft; the 18.04 debs are what everyone installs on jammy.
  # --- Do NOT install libsoundio1 first: jammy has no such package (only
  # --- libsoundio2), and libk4a1.4 does not depend on it anyway - its
  # --- Depends are libc6/libgcc1/libgl1/libstdc++6/libudev1/libx11-6, all
  # --- already present. libsoundio belongs to k4a-tools, not to the library.
  cd /tmp
  wget https://packages.microsoft.com/ubuntu/18.04/prod/pool/main/libk/libk4a1.4/libk4a1.4_1.4.1_amd64.deb
  wget https://packages.microsoft.com/ubuntu/18.04/prod/pool/main/libk/libk4a1.4-dev/libk4a1.4-dev_1.4.1_amd64.deb
  wget https://packages.microsoft.com/ubuntu/18.04/prod/pool/main/k/k4a-tools/k4a-tools_1.4.1_amd64.deb
  sudo apt install ./libk4a1.4_1.4.1_amd64.deb ./libk4a1.4-dev_1.4.1_amd64.deb ./k4a-tools_1.4.1_amd64.deb

  # --- udev rule, so the camera works without root
  sudo wget -O /etc/udev/rules.d/99-k4a.rules \\
    https://raw.githubusercontent.com/microsoft/Azure-Kinect-Sensor-SDK/develop/scripts/99-k4a.rules
  sudo udevadm control --reload-rules && sudo udevadm trigger
  # then UNPLUG AND REPLUG the camera

  # --- verify the SDK itself before touching python:
  k4aviewer""")
    if "binding" in missing:
        print("""
  # --- python binding (libk4a1.4-dev must already be installed: pyk4a compiles)
  uv pip install --python .venv/bin/python pyk4a
  # if that fails to build, the ctypes one needs no compiler:
  uv pip install --python .venv/bin/python pykinect-azure""")
    return 1


# ---------------------------------------------------------------------------
# cloud construction
# ---------------------------------------------------------------------------
def cloud_from_arrays(points_mm: np.ndarray, color_img: np.ndarray) -> Cloud:
    """(H,W,3) int16 mm + (H,W,3|4) BGR[A] uint8  ->  Cloud in metres.

    Invalid depth is z == 0 in k4a, and those pixels carry garbage x/y too, so
    they are dropped rather than kept as zeros.
    """
    if points_mm.shape[:2] != color_img.shape[:2]:
        raise ValueError(
            f"points {points_mm.shape[:2]} and color {color_img.shape[:2]} are different "
            "sizes - the two --align modes must not be mixed"
        )
    h, w = points_mm.shape[:2]
    pts = points_mm.reshape(-1, 3)
    valid = pts[:, 2] != 0

    xyz = pts[valid].astype(np.float32) / 1000.0  # mm -> m, the one conversion
    bgr = color_img.reshape(-1, color_img.shape[-1])[valid][:, :3]
    rgb = bgr[:, ::-1].copy().astype(np.uint8)  # k4a colour is BGRA
    return Cloud(xyz=xyz, rgb=rgb, src_shape=(h, w), n_raw=h * w)


def voxel_downsample(cloud: Cloud, voxel_m: float) -> Cloud:
    """One point per voxel (numpy only - open3d is not installed in .venv)."""
    if voxel_m <= 0 or len(cloud) == 0:
        return cloud
    keys = np.floor(cloud.xyz / voxel_m).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    idx.sort()
    return Cloud(cloud.xyz[idx], cloud.rgb[idx], cloud.src_shape, cloud.n_raw)


def describe(cloud: Cloud, tag: str = "cloud") -> None:
    """Print readings that can FAIL - a silent empty cloud is the failure mode."""
    n = len(cloud)
    pct = 100.0 * n / cloud.n_raw if cloud.n_raw else 0.0
    print(f"[{tag}] {n} points from {cloud.src_shape[1]}x{cloud.src_shape[0]} "
          f"({pct:.1f}% of pixels had valid depth)"
          f"{'   <-- EMPTY CLOUD' if n == 0 else ''}")
    if n == 0:
        return
    z = cloud.xyz[:, 2]
    print(f"[{tag}] depth z: min {z.min():.3f}  median {np.median(z):.3f}  "
          f"max {z.max():.3f} m"
          f"{'   <-- LOOKS LIKE MILLIMETRES, NOT METRES' if np.median(z) > 100 else ''}"
          f"{'   <-- ALL POINTS AT ONE DEPTH' if z.max() - z.min() < 1e-4 else ''}")
    lo, hi = cloud.xyz.min(axis=0), cloud.xyz.max(axis=0)
    print(f"[{tag}] bbox   : x [{lo[0]:+.3f},{hi[0]:+.3f}]  "
          f"y [{lo[1]:+.3f},{hi[1]:+.3f}]  z [{lo[2]:+.3f},{hi[2]:+.3f}] m "
          f"(depth-camera optical frame: x right, y down, z forward)")


# ---------------------------------------------------------------------------
# output
# ---------------------------------------------------------------------------
def write_ply(path: pathlib.Path, cloud: Cloud) -> None:
    """Binary little-endian PLY - readable by CloudCompare, MeshLab, open3d."""
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"comment generated by scripts/kinect_pointcloud.py\n"
        f"comment frame: azure kinect depth-camera optical (x right, y down, z forward)\n"
        f"comment units: metres\n"
        f"element vertex {len(cloud)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    ).encode("ascii")

    rec = np.empty(
        len(cloud),
        dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
               ("red", "u1"), ("green", "u1"), ("blue", "u1")],
    )
    rec["x"], rec["y"], rec["z"] = cloud.xyz.T
    rec["red"], rec["green"], rec["blue"] = cloud.rgb.T
    with open(path, "wb") as fh:
        fh.write(header)
        fh.write(rec.tobytes())
    print(f"[out] {path}  ({len(cloud)} points, {path.stat().st_size / 1e6:.1f} MB)")


def write_npz(path: pathlib.Path, cloud: Cloud) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, xyz=cloud.xyz, rgb=cloud.rgb)
    print(f"[out] {path}  ({len(cloud)} points, {path.stat().st_size / 1e6:.1f} MB)")


# ---------------------------------------------------------------------------
# capture (pyk4a)
# ---------------------------------------------------------------------------
def capture_clouds(frames: int, align: str, depth_mode: str, color_res: str, fps: int,
                   color_format: str = "COLOR_BGRA32", device_id: int = 0,
                   info: dict | None = None):
    """Yield (index, Cloud) from a live camera.  Importable for the recorder.

    device_id 选第几台(多相机);info 不为 None 时,start 之后写入
    info["serial"] —— 录制器用序列号给每台相机的数据目录命名(序列号不随
    插拔顺序变,device_id 会变)。"""
    import pyk4a
    from pyk4a import Config, PyK4A

    k4a = PyK4A(
        Config(
            color_resolution=getattr(pyk4a.ColorResolution, f"RES_{color_res}"),
            depth_mode=getattr(pyk4a.DepthMode, depth_mode),
            camera_fps=getattr(pyk4a.FPS, f"FPS_{fps}"),
            # PINNED, not left to the default: capture.transformed_color only
            # works for COLOR_BGRA32, and the whole --align depth path depends
            # on it. It IS the pyk4a default today, so relying on that would
            # work and then break silently the day anyone adds a format flag.
            color_format=getattr(pyk4a.ImageFormat, color_format),
            synchronized_images_only=True,  # never pair a depth frame with a stale colour one
        ),
        device_id=device_id,
    )
    k4a.start()
    if info is not None:
        info["serial"] = str(k4a.serial)
    print(f"[k4a] started  dev={device_id} serial={k4a.serial}  depth={depth_mode}  "
          f"color=RES_{color_res}  fps={fps}")
    try:
        got = 0
        dropped = 0
        while got < frames:
            capture = k4a.get_capture()
            if capture.depth is None or capture.color is None:
                dropped += 1
                if dropped > 30:
                    raise RuntimeError(
                        "30 captures in a row had no depth or no colour - is something "
                        "blocking the sensor, or is the cable on USB2?"
                    )
                continue
            if align == "depth":
                # points at DEPTH resolution, colour warped onto them.
                points_mm, color_img = capture.depth_point_cloud, capture.transformed_color
            else:
                # points at COLOUR resolution.  Bigger, and holes where depth is missing.
                points_mm, color_img = capture.transformed_depth_point_cloud, capture.color
            yield got, cloud_from_arrays(points_mm, color_img)
            got += 1
    finally:
        k4a.stop()
        print("[k4a] stopped")


# ---------------------------------------------------------------------------
# synthetic camera - so the VIEWER can be proven with no hardware attached
# ---------------------------------------------------------------------------
def mock_clouds(frames: int, fps: float = 15.0):
    """Yield (index, Cloud) for a scene whose geometry is KNOWN exactly.

    Back wall at z = 2.500 m, floor 0.800 m below the camera, and a sphere of
    radius 0.150 m orbiting at about 1.2 m. Nothing here touches the k4a SDK.
    It exists so the viewer, the reprojection, the units and the colour order
    can all be checked before the camera arrives - which leaves ONLY the camera
    and the SDK unproven, instead of the whole chain at once.

    Optical-frame convention throughout, same as the real thing: x right,
    y DOWN, z forward.
    """
    u = np.arange(_MOCK_W, dtype=np.float32)[None, :]
    v = np.arange(_MOCK_H, dtype=np.float32)[:, None]
    dx = np.broadcast_to((u - _MOCK_CX) / _MOCK_FX, (_MOCK_H, _MOCK_W))
    dy = np.broadcast_to((v - _MOCK_CY) / _MOCK_FY, (_MOCK_H, _MOCK_W))

    WALL_Z, FLOOR_Y, SPHERE_R = 2.5, 0.8, 0.15
    for i in range(frames):
        t = i / max(fps, 1e-6)
        z = np.full((_MOCK_H, _MOCK_W), WALL_Z, dtype=np.float32)
        col = np.zeros((_MOCK_H, _MOCK_W, 4), dtype=np.uint8)
        col[..., :3] = (170, 170, 170)                       # wall, BGR

        with np.errstate(divide="ignore", invalid="ignore"):
            z_floor = np.where(dy > 1e-6, FLOOR_Y / np.maximum(dy, 1e-6), np.inf)
        take = z_floor < z
        z = np.where(take, z_floor, z)
        chk = ((np.floor(dx * z * 4) + np.floor(z * 4)) % 2).astype(bool)
        col[..., :3] = np.where((take & chk)[..., None], (90, 90, 90),
                                np.where(take[..., None], (130, 130, 130),
                                         col[..., :3])).astype(np.uint8)

        c = np.array([0.45 * np.cos(1.2 * t), -0.05,
                      1.20 + 0.25 * np.sin(1.2 * t)], dtype=np.float32)
        a = dx * dx + dy * dy + 1.0
        b = -2.0 * (dx * c[0] + dy * c[1] + c[2])
        c0 = float(c @ c) - SPHERE_R ** 2
        disc = b * b - 4.0 * a * c0
        hit = disc > 0.0
        z_sph = np.where(hit, (-b - np.sqrt(np.maximum(disc, 0.0))) / (2.0 * a),
                         np.inf)
        take = hit & (z_sph > 0.0) & (z_sph < z)
        z = np.where(take, z_sph, z)
        col[..., :3] = np.where(take[..., None], (40, 140, 255),  # BGR -> orange
                                col[..., :3]).astype(np.uint8)

        pts_mm = np.stack([dx * z * 1000.0, dy * z * 1000.0, z * 1000.0],
                          axis=-1).astype(np.int16)
        yield i, cloud_from_arrays(pts_mm, col)


# ---------------------------------------------------------------------------
# live view (viser: browser, no GPU, same as every other viewer in this repo)
# ---------------------------------------------------------------------------
def live_view(args) -> int:
    import viser

    if args.source == "k4a":
        if importlib.util.find_spec("pyk4a") is None:
            print("[error] pyk4a is not installed. Run with --probe for the fix.\n"
                  "        To check the VIEWER meanwhile: --live --source mock",
                  file=sys.stderr)
            return 2
        src = capture_clouds(10 ** 9, args.align, args.depth_mode,
                             args.color_res, args.fps)
    else:
        src = mock_clouds(10 ** 9)

    server = viser.ViserServer(port=args.port)
    # The cloud is NOT rotated into any robot frame - it is exactly what the
    # camera produced. Only the viewer's idea of "up" is set, so that a y-DOWN
    # optical frame is not displayed upside down.
    server.scene.set_up_direction("-y")
    server.scene.add_grid("/grid", width=4.0, height=4.0)
    g_n = server.gui.add_text("点数", "-")
    g_z = server.gui.add_text("深度 z 中位", "-")
    g_fps = server.gui.add_text("帧率", "-")
    g_src = server.gui.add_text("来源", args.source)
    g_size = server.gui.add_slider("点大小", 0.001, 0.02, 0.001, args.point_size)
    server.gui.add_markdown(
        "坐标系为深度相机光学系(x 向右、y 向下、z 向前),并非机器人坐标系。\n\n"
        "深度 z 中位可以用卷尺核对:正对一面平整的墙,该数值应等于实测的距离,单位为米。"
        + ("\n\n当前为合成相机。场景中墙面距离 2.500 米,地板位于下方 0.800 米,"
           "球体半径 0.150 米。能看到这些即说明可视化、重投影、单位与颜色均正确,"
           "尚未验证的只有相机本身。" if args.source == "mock" else ""))

    print(f"[live] open http://localhost:{args.port}")
    t_last, n, t0 = time.perf_counter(), 0, time.perf_counter()
    try:
        for i, cloud in src:
            if args.voxel > 0:
                cloud = voxel_downsample(cloud, args.voxel)
            xyz, rgb = cloud.xyz, cloud.rgb
            if args.max_points and len(xyz) > args.max_points:
                step = int(np.ceil(len(xyz) / args.max_points))
                xyz, rgb = xyz[::step], rgb[::step]
            server.scene.add_point_cloud("/cloud", points=xyz, colors=rgb,
                                         point_size=float(g_size.value))
            n += 1
            now = time.perf_counter()
            g_n.value = f"{len(xyz)},抽稀前 {len(cloud)}"
            g_z.value = (f"{np.median(cloud.xyz[:, 2]):.3f} m" if len(cloud)
                         else "无有效点")
            g_fps.value = f"{1.0 / max(now - t_last, 1e-6):.1f} Hz"
            t_last = now
            if args.headless_seconds and now - t0 > args.headless_seconds:
                print(f"[live] 自检结束:共 {n} 帧,点数 {g_n.value},"
                      f"深度中位 {g_z.value},帧率 {g_fps.value}")
                return 0 if n > 0 else 1
    except KeyboardInterrupt:
        pass
    return 0


# ---------------------------------------------------------------------------
# self-check - known answer, no camera
# ---------------------------------------------------------------------------
def _synth_depth() -> tuple[np.ndarray, np.ndarray, dict]:
    """A depth image whose correct point cloud is known exactly.

    Left half   : fronto-parallel wall at 1.500 m
    Right half  : fronto-parallel wall at 2.000 m
    A 40x40 patch: depth 0 = invalid, must be dropped
    """
    depth_mm = np.zeros((_MOCK_H, _MOCK_W), dtype=np.uint16)
    depth_mm[:, : _MOCK_W // 2] = 1500
    depth_mm[:, _MOCK_W // 2 :] = 2000
    depth_mm[100:140, 100:140] = 0  # the hole

    u = np.arange(_MOCK_W, dtype=np.float32)[None, :]
    v = np.arange(_MOCK_H, dtype=np.float32)[:, None]
    z = depth_mm.astype(np.float32)
    points_mm = np.stack(
        [(u - _MOCK_CX) * z / _MOCK_FX, (v - _MOCK_CY) * z / _MOCK_FY, z], axis=-1
    ).astype(np.int16)

    color = np.zeros((_MOCK_H, _MOCK_W, 4), dtype=np.uint8)
    color[..., 0] = 10  # B
    color[..., 1] = 20  # G
    color[..., 2] = 30  # R  -> expect RGB (30, 20, 10) after the BGR flip
    truth = {"n_valid": _MOCK_H * _MOCK_W - 40 * 40, "near": 1.5, "far": 2.0}
    return points_mm, color, truth


def self_check(tmp: pathlib.Path) -> int:
    """Verify reprojection, units, invalid-pixel handling, colour order, PLY.

    NOT covered: the SDK's own transformation - that needs the real libk4a.
    """
    print("[self-check] synthetic depth image, known answer")
    points_mm, color, truth = _synth_depth()
    cloud = cloud_from_arrays(points_mm, color)
    describe(cloud, "self-check")

    fails: list[str] = []

    if len(cloud) != truth["n_valid"]:
        fails.append(f"point count {len(cloud)} != {truth['n_valid']} "
                     "(invalid depth pixels were not dropped)")

    z = cloud.xyz[:, 2]
    for want in (truth["near"], truth["far"]):
        got = z[np.isclose(z, want, atol=2e-3)]
        if got.size == 0:
            fails.append(f"no points at z={want:.3f} m - units are wrong "
                         f"(z ranges {z.min():.3f}..{z.max():.3f})")
    if np.abs(z).max() > 100:
        fails.append("z is in the hundreds - millimetres leaked through as metres")

    # x at the left edge of the near wall: (0 - cx) * 1.5 / fx
    want_x = (0.0 - _MOCK_CX) * truth["near"] / _MOCK_FX
    got_x = cloud.xyz[:, 0].min()
    if not np.isclose(got_x, want_x, atol=2e-3):
        fails.append(f"leftmost x {got_x:+.4f} != expected {want_x:+.4f} m (reprojection is wrong)")

    if not np.array_equal(cloud.rgb[0], np.array([30, 20, 10], dtype=np.uint8)):
        fails.append(f"colour {cloud.rgb[0].tolist()} != [30,20,10] - BGR/RGB is flipped")

    # voxel downsample must not move points or invent them
    small = voxel_downsample(cloud, 0.05)
    if len(small) >= len(cloud) or len(small) == 0:
        fails.append(f"voxel downsample produced {len(small)} from {len(cloud)}")

    # PLY round-trip
    ply = tmp / "self_check.ply"
    write_ply(ply, small)
    with open(ply, "rb") as fh:
        head = fh.read(400).decode("ascii", "replace")
    if f"element vertex {len(small)}" not in head:
        fails.append("PLY header vertex count does not match the cloud")
    body = ply.stat().st_size - (head.index("end_header\n") + len("end_header\n"))
    if body != len(small) * 15:  # 3*float32 + 3*uint8
        fails.append(f"PLY body is {body} bytes, expected {len(small) * 15}")

    if fails:
        print("\n[self-check] FAILED:")
        for f in fails:
            print(f"  - {f}")
        return 1
    print(f"[self-check] PASSED  ({len(cloud)} points, near/far walls at "
          f"{truth['near']}/{truth['far']} m recovered, PLY round-trip ok)")
    print("[self-check] NOTE: this exercises reprojection, units, colour order and "
          "the writer.\n             It does NOT exercise the k4a SDK - that needs the camera.")
    return 0


# ---------------------------------------------------------------------------
def record_session(args) -> int:
    """按页面的录制开关把点云写进本次录制的目录。

    这一路的难点不是接线是**带宽**:默认档整帧约 15 万点、30 fps,未压缩
    166 MB/s —— 按当前剩余磁盘只够录 46 分钟。所以录制默认降采样(体素 1cm)
    并降到 10 Hz,并且每次开录都把实测码率打出来,不让它悄悄把盘写满。

    存 .npz 追加不了,所以每帧一个文件放在 cloud/ 下,索引即帧号;时间戳单独
    存一份 cloud_t.msgpack,口径与其它各路一致(主机墙上时钟)。
    """
    import msgpack
    from magicdexmate.control_link import ControlSubscriber

    import threading

    ctl = ControlSubscriber(args.record_session)
    print(f"[k4a] 订阅开关频道 {args.record_session};录制体素 {args.rec_voxel} m、"
          f"{args.rec_hz:.0f} Hz")
    # 页面画面流:与录制无关,进程活着就发。页面因此能随时显示「相机拍到了
    # 什么」和连接状态,而不必自己打开设备(Kinect 独占,页面开了录制进程就
    # 打不开 —— 2026-08-11 用户就是被这一点卡住才有的这条流)。
    view_pub = None
    view_lock = threading.Lock()          # zmq socket 不是线程安全的
    if args.pub_view:
        import zmq as _zmq
        view_pub = _zmq.Context.instance().socket(_zmq.PUB)
        view_pub.setsockopt(_zmq.SNDHWM, 4)
        view_pub.bind(args.pub_view)
        print(f"[k4a] 画面流发布于 {args.pub_view}(每台约 5 Hz,体素 2cm)")

    # 多相机(2026-08-11 用户要求:接几台就录几台)。每台一个采集线程,身份
    # 用**序列号**(不随插拔顺序变;device_id 会变),数据各写各的目录:
    #   data/sessions/<会话>/cloud_<序列号>/NNNNNN.npz + cloud_t_<序列号>.msgpack
    # 录制开关是全体共享的一份状态,由主线程从页面广播里读。
    if args.source == "mock":
        cams = [("mock", i) for i in range(max(1, args.mock_cams))]
    else:
        import pyk4a
        n_dev = pyk4a.connected_device_count()
        if n_dev == 0:
            print("[k4a] 没有检测到任何相机")
            return 1
        cams = [("k4a", i) for i in range(n_dev)]
        print(f"[k4a] 检测到 {n_dev} 台相机,逐台启动")

    state = {"on": False, "session": "", "stop": False}
    N = 10 ** 6          # frames=0 在生成器里是「0 帧」不是「不限」

    def worker(kind: str, dev_id: int):
        info: dict = {}
        serial = f"MOCK{dev_id}" if kind == "mock" else ""
        gen = (mock_clouds(N, args.fps) if kind == "mock"
               else capture_clouds(N, args.align, args.depth_mode,
                                   args.color_res, args.fps,
                                   device_id=dev_id, info=info))
        d = None
        n = 0
        nbytes = 0
        t_last = 0.0
        view_last = 0.0
        t0_rec = 0.0
        tfh = None
        try:
            for _, cloud in gen:
                if state["stop"]:
                    break
                if not serial:
                    serial = info.get("serial", f"dev{dev_id}")
                now = time.time()
                if view_pub is not None and now - view_last >= 0.2:
                    view_last = now
                    vc = voxel_downsample(cloud, 0.02)
                    try:
                        with view_lock:
                            view_pub.send(msgpack.packb(
                                {"t_wall_us": int(now * 1e6),
                                 "serial": serial, "n": int(len(vc.xyz)),
                                 "xyz_mm": (vc.xyz * 1000.0)
                                 .astype(np.int16).tobytes(),
                                 "rgb": vc.rgb.astype(np.uint8).tobytes()},
                                use_bin_type=True), _zmq.NOBLOCK)
                    except Exception:                      # noqa: BLE001
                        pass
                on = state["on"]
                if on and d is None:
                    d = (ROOT / "data" / "sessions" / state["session"]
                         / f"cloud_{serial}")
                    d.mkdir(parents=True, exist_ok=True)
                    tfh = open(d.parent / f"cloud_t_{serial}.msgpack", "ab")
                    n, nbytes = 0, 0
                    print(f"\n[k4a:{serial}] 开始录点云 -> {d}")
                elif not on and d is not None:
                    secs = max(1e-6, now - t0_rec)
                    hz = n / secs
                    warn = ("   <-- 低于设定,瓶颈在取点云而不是写盘;"
                            "--rec-hz 是上限,不能让源变快"
                            if hz < 0.7 * args.rec_hz else "")
                    print(f"\n[k4a:{serial}] 停止,共 {n} 帧 / {nbytes/1e6:.1f} MB "
                          f"({nbytes/1e6/secs:.1f} MB/s,实测 {hz:.1f} Hz"
                          f"/设定 {args.rec_hz:.0f} Hz){warn}")
                    tfh.close()
                    tfh = None
                    d = None
                    continue
                if d is None:
                    continue
                if n == 0:
                    t0_rec = now
                if now - t_last < 1.0 / max(args.rec_hz, 1e-6):
                    continue
                t_last = now
                if args.rec_voxel > 0:
                    cloud = voxel_downsample(cloud, args.rec_voxel)
                xyz, rgb = cloud.xyz, cloud.rgb
                # int16 毫米、不压缩。实测(9.3 万点):压缩 float32 是 61 ms/帧、
                # 1.29 MB;int16 毫米不压缩是 1.0 ms/帧、0.84 MB —— 又快 60 倍又
                # 更小。毫米是 k4a 的原生单位,不是精度损失。**单位写进文件**:
                # 这个仓库全线用米,毫米当米读就是 1000 倍的静默错误。
                f = d / f"{n:06d}.npz"
                np.savez(f, xyz_mm=(xyz * 1000.0).astype(np.int16),
                         rgb=rgb.astype(np.uint8), unit=np.bytes_(b"millimetre"))
                nbytes += f.stat().st_size
                tfh.write(msgpack.packb({"t_wall_us": int(now * 1e6), "i": n,
                                         "n_points": int(len(xyz))},
                                        use_bin_type=True))
                n += 1
        except Exception as e:                             # noqa: BLE001
            # 一台相机死了不拖累别的相机;打清楚是谁死的、为什么。
            print(f"\n[k4a:{serial or dev_id}] 采集线程退出:{type(e).__name__} {e}")
        finally:
            if tfh is not None:
                tfh.close()

    threads = [threading.Thread(target=worker, args=c, daemon=True)
               for c in cams]
    for th in threads:
        th.start()
    try:
        while any(th.is_alive() for th in threads):
            st = ctl.latest(time.time()) or {}
            state["on"] = bool(st.get("recording") and st.get("session"))
            if state["on"]:
                state["session"] = str(st["session"])
            time.sleep(0.02)
    except KeyboardInterrupt:
        pass
    finally:
        state["stop"] = True
        for th in threads:
            th.join(timeout=5)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--probe", action="store_true",
                    help="report SDK/binding/device status and how to fix it, then exit")
    ap.add_argument("--self-check", action="store_true",
                    help="verify the maths on a synthetic frame (no camera), then exit")
    ap.add_argument("--frames", type=int, default=1,
                    help="how many captures to take (default 1; only the last is written)")
    ap.add_argument("--align", choices=("depth", "color"), default="depth",
                    help="depth: points at depth res with colour warped on (default). "
                         "color: points at colour res, has holes")
    ap.add_argument("--depth-mode", default="NFOV_UNBINNED",
                    choices=("NFOV_UNBINNED", "NFOV_2X2BINNED", "WFOV_UNBINNED", "WFOV_2X2BINNED"))
    ap.add_argument("--color-res", default="720P",
                    choices=("720P", "1080P", "1440P", "1536P", "2160P", "3072P"))
    ap.add_argument("--fps", type=int, default=30, choices=(5, 15, 30))
    ap.add_argument("--voxel", type=float, default=0.0,
                    help="voxel downsample size in metres (0 = off)")
    ap.add_argument("--out", type=pathlib.Path, default=None,
                    help="write a .ply or .npz")
    ap.add_argument("--binding", choices=("pyk4a",), default="pyk4a",
                    help="pykinectazure is a ctypes fallback if pyk4a will not build; "
                         "not wired up yet")
    ap.add_argument("--live", action="store_true",
                    help="实时点云,在浏览器中查看")
    ap.add_argument("--source", choices=("k4a", "mock"), default="k4a",
                    help="mock 为合成相机,不需要硬件,用于验证可视化部分")
    ap.add_argument("--port", type=int, default=8087,
                    help="浏览器端口。8080 至 8086 已被本仓库其它查看器占用")
    ap.add_argument("--max-points", type=int, default=120000,
                    help="送入浏览器的点数上限。整帧 640x576 约 36.8 万点,过多会导致页面卡顿")
    ap.add_argument("--point-size", type=float, default=0.006)
    ap.add_argument("--record-session", metavar="ADDR", default="",
                    help="订开关频道(例如 tcp://127.0.0.1:5584),按页面的录制"
                         "开关把点云写进 data/sessions/<session>/cloud.npz。"
                         "默认降采样到 --rec-voxel / --rec-hz,见下")
    ap.add_argument("--rec-voxel", type=float, default=0.01,
                    help="录制时的体素边长,单位米。0 = 不降采样。默认 1cm:"
                         "整帧 15 万点未降采样是 166 MB/s,磁盘只够录 46 分钟")
    ap.add_argument("--mock-cams", type=int, default=1,
                    help="mock 模式模拟几台相机(多相机录制的免硬件测试用)")
    ap.add_argument("--pub-view", metavar="ADDR", default="",
                    help="把降采样点云(约 5Hz)发布到此 zmq 地址,供主页面"
                         "(viser_isaac_mirror)显示相机画面与连接状态。与录制"
                         "开关无关,进程活着就一直发。控制台传 tcp://*:5591")
    ap.add_argument("--rec-hz", type=float, default=10.0,
                    help="录制帧率。相机出 30 fps,点云录制降到 10 已经够用")
    ap.add_argument("--headless-seconds", type=float, default=0.0,
                    help="无界面自检:运行指定秒数后打印读数并退出")
    args = ap.parse_args()

    if args.record_session:
        return record_session(args)
    if args.probe:
        return probe()
    if args.self_check:
        return self_check(pathlib.Path("/tmp"))
    if args.live:
        return live_view(args)

    if importlib.util.find_spec("pyk4a") is None:
        print("[error] pyk4a is not installed. Run with --probe for the fix.", file=sys.stderr)
        return 2

    last: Cloud | None = None
    t0 = time.perf_counter()
    for i, cloud in capture_clouds(args.frames, args.align, args.depth_mode,
                                   args.color_res, args.fps):
        if args.voxel > 0:
            cloud = voxel_downsample(cloud, args.voxel)
        describe(cloud, f"frame {i}")
        last = cloud
    dt = time.perf_counter() - t0
    print(f"[rate] {args.frames} frames in {dt:.2f}s = {args.frames / dt:.1f} Hz")

    if last is None:
        print("[error] no frames captured", file=sys.stderr)
        return 1
    if args.out:
        if args.out.suffix == ".npz":
            write_npz(args.out, last)
        else:
            write_ply(args.out, last)
    else:
        print("[out] nothing written (pass --out /tmp/cloud.ply)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
