from __future__ import annotations

import shlex
import signal
import socket
import subprocess
import time
from dataclasses import dataclass

NANO_TARGET = "dexmate-nano@192.168.50.22"
THOR_TARGET = "dexmate@192.168.50.20"
STREAMER_BINARY = "/home/dexmate-nano/zed_stream/build/zed_streamer"
STREAMER_ARGUMENTS = (
    "--jpeg-quality",
    "100",
    "--max-fps",
    "30",
    "--resolution",
    "HD1200",
    "--no-right",
    "--no-depth",
    "--no-pc",
    "--no-imu",
)


@dataclass(frozen=True)
class SSHRoute:
    name: str
    proxy_command: str | None = None


class RemoteStreamerManager:
    """Keep one SSH session attached to a fixed HD1200 streamer command."""

    def __init__(
        self,
        *,
        nano_target: str = NANO_TARGET,
        thor_target: str = THOR_TARGET,
        route_preference: str = "auto",
        ssh_timeout_s: int = 5,
        use_sudo: bool = True,
    ) -> None:
        if route_preference not in {"auto", "direct", "proxy"}:
            raise ValueError("route_preference must be auto, direct, or proxy")
        self.nano_target = nano_target
        self.thor_target = thor_target
        self.route_preference = route_preference
        self.ssh_timeout_s = ssh_timeout_s
        self.use_sudo = use_sudo
        self.route: SSHRoute | None = None
        self.process: subprocess.Popen[bytes] | None = None

    @property
    def direct_route(self) -> SSHRoute:
        return SSHRoute("direct")

    @property
    def proxy_route(self) -> SSHRoute:
        return SSHRoute(
            "proxy",
            f"ssh -o BatchMode=yes -o ConnectTimeout={self.ssh_timeout_s} "
            f"-W %h:%p {self.thor_target}",
        )

    def _ssh_prefix(self, route: SSHRoute, *, allocate_tty: bool) -> list[str]:
        command = [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            f"ConnectTimeout={self.ssh_timeout_s}",
        ]
        if allocate_tty:
            command.append("-tt")
        if route.proxy_command is not None:
            command.extend(["-o", f"ProxyCommand={route.proxy_command}"])
        command.append(self.nano_target)
        return command

    def _probe(self, route: SSHRoute) -> subprocess.CompletedProcess[str]:
        command = [*self._ssh_prefix(route, allocate_tty=False), "true"]
        try:
            return subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=self.ssh_timeout_s + 2,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RuntimeError(f"SSH {route.name} probe failed: {exc}") from exc

    def select_route(self) -> SSHRoute:
        if self.route is not None:
            return self.route
        if self.route_preference == "direct":
            candidates = [self.direct_route]
        elif self.route_preference == "proxy":
            candidates = [self.proxy_route]
        else:
            candidates = [self.direct_route, self.proxy_route]

        failures = []
        for route in candidates:
            result = self._probe(route)
            if result.returncode == 0:
                self.route = route
                return route
            failures.append(f"{route.name}: {result.stderr.strip() or 'SSH failed'}")
        raise RuntimeError(
            "Cannot authenticate to the camera Nano with BatchMode SSH. "
            "Install this Mac's public key on dexmate-nano@192.168.50.22. " + " | ".join(failures)
        )

    @staticmethod
    def port_is_open(host: str, port: int, timeout_s: float = 0.5) -> bool:
        try:
            with socket.create_connection((host, port), timeout=timeout_s):
                return True
        except OSError:
            return False

    def _remote_streamer_command(self) -> str:
        command = [STREAMER_BINARY, *STREAMER_ARGUMENTS]
        if self.use_sudo:
            command = ["sudo", *command]
        # exec keeps the streamer as the foreground process attached to this SSH PTY.
        return f"cd /home/dexmate-nano/zed_stream && exec {shlex.join(command)}"

    def start_attached(self) -> None:
        if self.process is not None:
            raise RuntimeError("A managed streamer SSH session is already running")
        route = self.select_route()
        command = [
            *self._ssh_prefix(route, allocate_tty=True),
            self._remote_streamer_command(),
        ]
        try:
            self.process = subprocess.Popen(command)
        except OSError as exc:
            raise RuntimeError(f"Could not start attached streamer SSH session: {exc}") from exc

    def wait_until_ready(self, host: str, port: int, timeout_s: float = 30.0) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.port_is_open(host, port):
                return
            if self.process is not None and self.process.poll() is not None:
                code = self.process.returncode
                self.process = None
                raise RuntimeError(
                    f"Remote streamer SSH session exited before opening {host}:{port} "
                    f"(exit {code}). Check the visible Nano output above."
                )
            time.sleep(0.25)
        raise RuntimeError(
            f"Remote streamer did not open {host}:{port} within {timeout_s:.1f}s. "
            "If another ZED process owns the camera, stop it explicitly; quickstart never uses --clean."
        )

    def ensure_started(self, host: str, port: int, timeout_s: float = 30.0) -> bool:
        """Return True only if this invocation opened an attached SSH process."""
        self.select_route()
        if self.port_is_open(host, port):
            return False
        self.start_attached()
        self.wait_until_ready(host, port, timeout_s)
        return True

    def stop_attached(
        self,
        *,
        host: str | None = None,
        port: int | None = None,
        timeout_s: float = 10.0,
    ) -> None:
        """Close only the SSH session launched by this manager."""
        process = self.process
        if process is None:
            return
        if process.poll() is None:
            process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=3.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=3.0)
        self.process = None
        if host is not None and port is not None:
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline and self.port_is_open(host, port):
                time.sleep(0.25)
            if self.port_is_open(host, port):
                raise RuntimeError(
                    f"SSH session closed but {host}:{port} is still open. "
                    "The streamer may have detached unexpectedly; stop it manually on Nano."
                )
