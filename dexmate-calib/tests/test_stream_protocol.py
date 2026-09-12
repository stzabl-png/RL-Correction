from __future__ import annotations

import io
import struct

import pytest

from dexmate_calib.streaming.zed_stream import (
    ENC_JPEG,
    ENC_LZ4,
    MAGIC_V1,
    MAGIC_V2,
    TYPE_DEPTH,
    TYPE_LEFT,
    StreamProtocolError,
    ZedStreamClient,
    read_frame,
)


def segment(seg_type: int, encoding: int, width: int, height: int, payload: bytes) -> bytes:
    raw_size = width * height * (4 if seg_type == TYPE_DEPTH else 1)
    return (
        struct.pack("<BBIIII", seg_type, encoding, width, height, len(payload), raw_size) + payload
    )


def frame_v1(*segments: bytes) -> bytes:
    return struct.pack("<4sQHH", MAGIC_V1, 123456789, 0x0005, len(segments)) + b"".join(segments)


def frame_v2(*segments: bytes, serial: int = 59595115) -> bytes:
    return struct.pack("<4sIQHH", MAGIC_V2, serial, 987654321, 0x0005, len(segments)) + b"".join(
        segments
    )


def test_reads_zs01_left_and_skips_depth() -> None:
    packet = frame_v1(
        segment(TYPE_LEFT, ENC_JPEG, 1920, 1200, b"jpeg-data"),
        segment(TYPE_DEPTH, ENC_LZ4, 1920, 1200, b"compressed-depth"),
    )
    parsed = read_frame(io.BytesIO(packet))
    assert parsed.protocol == "ZS01"
    assert parsed.serial_number is None
    assert parsed.left_jpeg == b"jpeg-data"
    assert (parsed.left_width, parsed.left_height) == (1920, 1200)
    assert [item.type for item in parsed.segments] == [TYPE_LEFT, TYPE_DEPTH]


def test_reads_zs02_serial() -> None:
    packet = frame_v2(segment(TYPE_LEFT, ENC_JPEG, 1920, 1200, b"jpeg"))
    parsed = read_frame(io.BytesIO(packet))
    assert parsed.protocol == "ZS02"
    assert parsed.serial_number == 59595115
    assert parsed.source_timestamp_ns == 987654321


def test_rejects_unknown_magic() -> None:
    with pytest.raises(StreamProtocolError, match="Unsupported frame magic"):
        read_frame(io.BytesIO(b"BAD!"))


def test_rejects_truncated_payload() -> None:
    packet = frame_v2(segment(TYPE_LEFT, ENC_JPEG, 1920, 1200, b"jpeg"))[:-2]
    with pytest.raises(ConnectionError, match="remaining"):
        read_frame(io.BytesIO(packet))


def test_client_locks_hd1200_and_serial() -> None:
    parsed = read_frame(io.BytesIO(frame_v2(segment(TYPE_LEFT, ENC_JPEG, 1920, 1200, b"jpeg"))))
    client = ZedStreamClient(expected_serial=59595115)
    client._validate_frame(parsed)
    assert client._locked_identity == ("ZS02", 59595115, 1920, 1200)


def test_client_rejects_hd1080() -> None:
    parsed = read_frame(io.BytesIO(frame_v2(segment(TYPE_LEFT, ENC_JPEG, 1920, 1080, b"jpeg"))))
    client = ZedStreamClient(expected_serial=59595115)
    with pytest.raises(StreamProtocolError, match="Wrong stream height"):
        client._validate_frame(parsed)
