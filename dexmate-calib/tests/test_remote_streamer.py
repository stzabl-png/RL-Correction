from __future__ import annotations

import signal
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import dexmate_calib.cli as cli_module
from dexmate_calib.cli import _cmd_intrinsics, build_parser
from dexmate_calib.remote.streamer import RemoteStreamerManager, SSHRoute


def completed(code: int, stderr: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(["ssh"], code, stdout="", stderr=stderr)


def test_fixed_remote_command_is_hd1200_left_only_without_clean() -> None:
    manager = RemoteStreamerManager()
    command = manager._remote_streamer_command()
    assert "sudo" in command
    assert "--resolution HD1200" in command
    assert "--no-right" in command
    assert "--no-depth" in command
    assert "--no-pc" in command
    assert "--no-imu" in command
    assert "--clean" not in command
    assert "systemctl" not in command


def test_no_sudo_mode_is_supported() -> None:
    command = RemoteStreamerManager(use_sudo=False)._remote_streamer_command()
    assert "sudo" not in command
    assert "exec /home/dexmate-nano/zed_stream/build/zed_streamer" in command


def test_auto_route_falls_back_to_thor_proxy(monkeypatch) -> None:
    manager = RemoteStreamerManager()

    def fake_probe(route: SSHRoute):
        return completed(0) if route.name == "proxy" else completed(255, "direct failed")

    monkeypatch.setattr(manager, "_probe", fake_probe)
    route = manager.select_route()
    assert route.name == "proxy"
    assert route.proxy_command is not None
    assert "BatchMode=yes" in route.proxy_command
    assert "-W %h:%p dexmate@192.168.50.20" in route.proxy_command


def test_existing_stream_is_not_started_or_owned(monkeypatch) -> None:
    manager = RemoteStreamerManager(route_preference="direct")
    monkeypatch.setattr(manager, "select_route", lambda: manager.direct_route)
    monkeypatch.setattr(manager, "port_is_open", lambda *_args, **_kwargs: True)
    start = Mock()
    monkeypatch.setattr(manager, "start_attached", start)
    assert manager.ensure_started("192.168.50.22", 30000) is False
    start.assert_not_called()


def test_attached_start_uses_tty_and_keeps_local_process(monkeypatch) -> None:
    manager = RemoteStreamerManager(route_preference="direct")
    manager.route = manager.direct_route
    child = Mock()
    popen = Mock(return_value=child)
    monkeypatch.setattr(subprocess, "Popen", popen)
    manager.start_attached()
    command = popen.call_args.args[0]
    assert "-tt" in command
    assert "BatchMode=yes" in command
    assert "--resolution HD1200" in command[-1]
    assert manager.process is child


def test_stop_signals_only_managed_ssh_process() -> None:
    manager = RemoteStreamerManager()
    child = Mock()
    child.poll.return_value = None
    child.wait.return_value = 0
    manager.process = child
    manager.stop_attached()
    child.send_signal.assert_called_once_with(signal.SIGINT)
    assert manager.process is None


def test_quickstart_cli_defaults_to_safe_fixed_profile() -> None:
    args = build_parser().parse_args(["intrinsics", "quickstart"])
    assert args.board == "dexmate-10x7"
    assert args.host == "192.168.50.22"
    assert args.port == 30000
    assert args.expected_serial == 59595115
    assert args.ssh_route == "auto"
    assert args.no_sudo is False
    assert args.min_grid_rows == 3
    assert args.min_grid_cols == 3
    assert args.min_board_bbox_fraction == 0.12
    assert args.detect_fps == 10.0
    assert args.no_solve is False


def _mock_quickstart_dependencies(monkeypatch, events: list[str]) -> Mock:
    manager = Mock()
    manager.process = object()
    manager.ensure_started.return_value = True
    manager.select_route.return_value = SimpleNamespace(name="direct")
    manager.stop_attached.side_effect = lambda **_kwargs: events.append("stop")

    monkeypatch.setattr(cli_module, "RemoteStreamerManager", Mock(return_value=manager))
    monkeypatch.setattr(cli_module, "resolve_board_profile", Mock(return_value=object()))
    monkeypatch.setattr(cli_module, "ZedStreamClient", Mock(return_value=object()))
    monkeypatch.setattr(cli_module, "doctor_stream", Mock(return_value={}))
    monkeypatch.setattr(cli_module, "print_report", Mock())
    monkeypatch.setattr(
        cli_module,
        "capture_session",
        Mock(side_effect=lambda *_args, **_kwargs: events.append("capture") or Path("session")),
    )
    return manager


def test_quickstart_solves_by_default_after_stopping_streamer(monkeypatch) -> None:
    events: list[str] = []
    _mock_quickstart_dependencies(monkeypatch, events)
    solve = Mock(side_effect=lambda *_args, **_kwargs: events.append("solve") or Path("result"))
    monkeypatch.setattr(cli_module, "solve_session", solve)

    args = build_parser().parse_args(["intrinsics", "quickstart"])
    assert _cmd_intrinsics(args) == 0

    solve.assert_called_once_with(Path("session"))
    assert events == ["capture", "stop", "solve"]


def test_quickstart_no_solve_stops_after_capture(monkeypatch) -> None:
    events: list[str] = []
    _mock_quickstart_dependencies(monkeypatch, events)
    solve = Mock()
    monkeypatch.setattr(cli_module, "solve_session", solve)

    args = build_parser().parse_args(["intrinsics", "quickstart", "--no-solve"])
    assert _cmd_intrinsics(args) == 0

    solve.assert_not_called()
    assert events == ["capture", "stop"]


def test_manual_capture_requires_preview() -> None:
    args = build_parser().parse_args(["intrinsics", "capture", "--manual", "--no-preview"])
    with pytest.raises(ValueError, match="preview window"):
        _cmd_intrinsics(args)


def test_solver_cli_defaults_to_deterministic_validation() -> None:
    args = build_parser().parse_args(["intrinsics", "solve", "session"])
    assert args.max_views == 40
    assert args.cross_validation_folds == 5
