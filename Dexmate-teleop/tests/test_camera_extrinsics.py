"""magicdexmate.camera_extrinsics: loader for RobotCamCalib/extr_calib_vega_kinect.py output."""
import numpy as np
import pytest
import yaml

from magicdexmate import camera_extrinsics as ce


def _T(rot_deg_z=30.0, t=(1.9, -0.4, 1.35)):
    c, s = np.cos(np.radians(rot_deg_z)), np.sin(np.radians(rot_deg_z))
    T = np.eye(4)
    T[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    T[:3, 3] = t
    return T


def _write(tmp_path, **override):
    T_bc = _T()
    T_cd = np.eye(4); T_cd[:3, 3] = [-0.032, 0.0, 0.002]
    d = {
        "schema": ce.SCHEMA, "camera": {"type": "azure_kinect", "serial": "000123"},
        "n_samples": 12, "K_color": np.diag([915.0, 914.0, 1.0]).tolist(),
        "dist_color": [0.05, -0.02, 0.0005, -0.0003, 0.001, 0.0, 0.0, 0.0],
        "T_base_color": T_bc.tolist(), "T_color_depth": T_cd.tolist(),
        "T_base_depth": (T_bc @ T_cd).tolist(),
    }
    d.update(override)
    p = tmp_path / "kinect_000123.yaml"
    p.write_text(yaml.safe_dump(d, sort_keys=False))
    return p, T_bc, T_cd


def test_load_by_path_and_by_serial(tmp_path, monkeypatch):
    p, T_bc, T_cd = _write(tmp_path)
    ext = ce.load(p)
    assert np.allclose(ext["T_base_color"], T_bc)
    assert ext["T_base_depth"].shape == (4, 4) and np.allclose(ext["T_base_depth"], T_bc @ T_cd)
    assert ext["dist_color"].shape == (8,)
    monkeypatch.setattr(ce, "DIR", tmp_path)
    assert np.allclose(ce.load("000123")["T_base_color"], T_bc)
    assert ce.available() == {"000123": p}


def test_transform_points_matches_matrix():
    T = _T()
    pts = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
    out = ce.transform_points(T, pts)
    assert np.allclose(out[0], T[:3, 3])
    assert np.allclose(out[1], (T @ np.append(pts[1], 1.0))[:3])


def test_rejects_wrong_schema_and_bad_rotation(tmp_path):
    p, _, _ = _write(tmp_path, schema="something.else")
    with pytest.raises(ValueError):
        ce.load(p)
    bad = np.eye(4); bad[0, 0] = 2.0
    p, _, _ = _write(tmp_path, T_base_color=bad.tolist())
    with pytest.raises(ValueError):
        ce.load(p)
    with pytest.raises(FileNotFoundError):
        ce.load(tmp_path / "kinect_nope.yaml")
