from __future__ import annotations

import socket
import struct
import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import BinaryIO

FRAME_V1_REST = struct.Struct("<QHH")
FRAME_V2_REST = struct.Struct("<IQHH")
SEGMENT_HEADER = struct.Struct("<BBIIII")

MAGIC_V1 = b"ZS01"
MAGIC_V2 = b"ZS02"
SUPPORTED_MAGICS = {MAGIC_V1, MAGIC_V2}

TYPE_LEFT = 0
TYPE_RIGHT = 1
TYPE_DEPTH = 2
TYPE_POINT_CLOUD = 3
TYPE_IMU = 4

ENC_JPEG = 0
ENC_LZ4 = 1
ENC_RAW = 2

TYPE_NAMES = {
    TYPE_LEFT: "left",
    TYPE_RIGHT: "right",
    TYPE_DEPTH: "depth",
    TYPE_POINT_CLOUD: "point_cloud",
    TYPE_IMU: "imu",
}
ENCODING_NAMES = {ENC_JPEG: "jpeg", ENC_LZ4: "lz4", ENC_RAW: "raw"}

MAX_SEGMENTS = 16
MAX_SEGMENT_BYTES = 256 * 1024 * 1024


class StreamProtocolError(RuntimeError):
    """The TCP stream does not conform to ZS01/ZS02."""


@dataclass(frozen=True)
class SegmentInfo:
    type: int
    encoding: int
    dim0: int
    dim1: int
    compressed_size: int
    raw_size: int

    @property
    def name(self) -> str:
        return TYPE_NAMES.get(self.type, f"unknown_{self.type}")


@dataclass(frozen=True)
class StreamFrame:
    protocol: str
    serial_number: int | None
    source_timestamp_ns: int
    receive_timestamp_ns: int
    channel_mask: int
    segments: tuple[SegmentInfo, ...]
    left_jpeg: bytes | None
    left_width: int | None
    left_height: int | None

    def decode_left_bgr(self):
        if self.left_jpeg is None:
            raise StreamProtocolError("Frame does not contain a left JPEG segment")
        try:
            import cv2
            import numpy as np
        except ImportError as exc:  # pragma: no cover - dependency error path
            raise RuntimeError("OpenCV and NumPy are required to decode JPEG") from exc
        image = cv2.imdecode(np.frombuffer(self.left_jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise StreamProtocolError("OpenCV failed to decode left JPEG")
        height, width = image.shape[:2]
        if self.left_width is not None and (width, height) != (
            self.left_width,
            self.left_height,
        ):
            raise StreamProtocolError(
                f"JPEG dimensions {width}x{height} disagree with header "
                f"{self.left_width}x{self.left_height}"
            )
        return image


def _read_exact(stream: BinaryIO | socket.socket, size: int) -> bytes:
    if size < 0:
        raise ValueError("size must be non-negative")
    buffer = bytearray(size)
    view = memoryview(buffer)
    offset = 0
    while offset < size:
        if hasattr(stream, "recv_into"):
            count = stream.recv_into(view[offset:], size - offset)  # type: ignore[attr-defined]
        else:
            chunk = stream.read(size - offset)
            count = len(chunk)
            view[offset : offset + count] = chunk
        if not count:
            raise ConnectionError(f"Stream closed with {size - offset} bytes remaining")
        offset += count
    return bytes(buffer)


def _discard_exact(stream: BinaryIO | socket.socket, size: int) -> None:
    remaining = size
    scratch = bytearray(min(1024 * 1024, max(1, size)))
    view = memoryview(scratch)
    while remaining:
        amount = min(remaining, len(scratch))
        if hasattr(stream, "recv_into"):
            count = stream.recv_into(view[:amount], amount)  # type: ignore[attr-defined]
        else:
            chunk = stream.read(amount)
            count = len(chunk)
        if not count:
            raise ConnectionError(f"Stream closed with {remaining} payload bytes remaining")
        remaining -= count


def read_frame(stream: BinaryIO | socket.socket) -> StreamFrame:
    magic = _read_exact(stream, 4)
    if magic == MAGIC_V1:
        timestamp_ns, channel_mask, n_segments = FRAME_V1_REST.unpack(
            _read_exact(stream, FRAME_V1_REST.size)
        )
        serial_number = None
    elif magic == MAGIC_V2:
        serial_number, timestamp_ns, channel_mask, n_segments = FRAME_V2_REST.unpack(
            _read_exact(stream, FRAME_V2_REST.size)
        )
    else:
        raise StreamProtocolError(f"Unsupported frame magic: {magic!r}")

    if n_segments > MAX_SEGMENTS:
        raise StreamProtocolError(f"Unreasonable segment count: {n_segments}")

    segments: list[SegmentInfo] = []
    left_jpeg: bytes | None = None
    left_width: int | None = None
    left_height: int | None = None
    for _ in range(n_segments):
        fields = SEGMENT_HEADER.unpack(_read_exact(stream, SEGMENT_HEADER.size))
        segment = SegmentInfo(*fields)
        if segment.compressed_size > MAX_SEGMENT_BYTES or segment.raw_size > MAX_SEGMENT_BYTES:
            raise StreamProtocolError(
                f"Segment {segment.name} size exceeds safety limit: "
                f"compressed={segment.compressed_size}, raw={segment.raw_size}"
            )
        segments.append(segment)
        if segment.type == TYPE_LEFT and segment.encoding == ENC_JPEG:
            if left_jpeg is not None:
                raise StreamProtocolError("Frame contains multiple left JPEG segments")
            left_jpeg = _read_exact(stream, segment.compressed_size)
            left_width = segment.dim0
            left_height = segment.dim1
        else:
            _discard_exact(stream, segment.compressed_size)

    return StreamFrame(
        protocol=magic.decode("ascii"),
        serial_number=serial_number,
        source_timestamp_ns=timestamp_ns,
        receive_timestamp_ns=time.time_ns(),
        channel_mask=channel_mask,
        segments=tuple(segments),
        left_jpeg=left_jpeg,
        left_width=left_width,
        left_height=left_height,
    )


class ZedStreamClient:
    def __init__(
        self,
        host: str = "192.168.50.22",
        port: int = 30000,
        *,
        connect_timeout_s: float = 5.0,
        read_timeout_s: float = 10.0,
        expected_serial: int | None = None,
        expected_width: int | None = 1920,
        expected_height: int | None = 1200,
    ) -> None:
        self.host = host
        self.port = port
        self.connect_timeout_s = connect_timeout_s
        self.read_timeout_s = read_timeout_s
        self.expected_serial = expected_serial
        self.expected_width = expected_width
        self.expected_height = expected_height
        self._socket: socket.socket | None = None
        self._locked_identity: tuple[str, int | None, int, int] | None = None

    def connect(self) -> None:
        if self._socket is not None:
            return
        sock = socket.create_connection((self.host, self.port), timeout=self.connect_timeout_s)
        sock.settimeout(self.read_timeout_s)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._socket = sock

    def close(self) -> None:
        if self._socket is not None:
            try:
                self._socket.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            self._socket.close()
            self._socket = None

    def __enter__(self) -> ZedStreamClient:  # noqa: PYI034
        self.connect()
        return self

    def __exit__(self, *_args) -> None:
        self.close()

    def receive(self) -> StreamFrame:
        self.connect()
        assert self._socket is not None
        frame = read_frame(self._socket)
        self._validate_frame(frame)
        return frame

    def frames(self) -> Iterator[StreamFrame]:
        while True:
            yield self.receive()

    def _validate_frame(self, frame: StreamFrame) -> None:
        if frame.left_jpeg is None:
            raise StreamProtocolError("Frame is missing required left JPEG channel")
        assert frame.left_width is not None and frame.left_height is not None
        if self.expected_serial is not None:
            if frame.serial_number is None:
                raise StreamProtocolError("Expected a camera serial, but ZS01 does not provide one")
            if frame.serial_number != self.expected_serial:
                raise StreamProtocolError(
                    f"Camera serial changed: expected {self.expected_serial}, "
                    f"received {frame.serial_number}"
                )
        if self.expected_width is not None and frame.left_width != self.expected_width:
            raise StreamProtocolError(
                f"Wrong stream width: expected {self.expected_width}, got {frame.left_width}"
            )
        if self.expected_height is not None and frame.left_height != self.expected_height:
            raise StreamProtocolError(
                f"Wrong stream height: expected {self.expected_height}, got {frame.left_height}"
            )
        identity = (
            frame.protocol,
            frame.serial_number,
            frame.left_width,
            frame.left_height,
        )
        if self._locked_identity is None:
            self._locked_identity = identity
        elif identity != self._locked_identity:
            raise StreamProtocolError(
                f"Stream identity changed within connection: {self._locked_identity} -> {identity}"
            )
