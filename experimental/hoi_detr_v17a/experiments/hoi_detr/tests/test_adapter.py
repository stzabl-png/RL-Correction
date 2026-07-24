"""Contract tests for the pure-Python HOI-DETR JSON adapter."""

from __future__ import annotations

import copy
import json
import math
import tempfile
import unittest
from pathlib import Path

from experiments.hoi_detr import (
    REPORT_SCHEMA_VERSION,
    SCHEMA_VERSION,
    CandidateValidationError,
    build_report,
    dumps_candidates,
    loads_candidates,
    normalize_official_video,
    read_candidates_json,
    read_official_video_json,
    summarize_candidates,
    validate_candidates,
    write_candidates_json,
    write_json_atomic,
    write_report_json,
)


def _official_payload():
    detections = [
        {"box": [1, 2, 21, 32], "score": 0.97, "class_id": 0, "class_name": "hand"},
        {"box": [20, 10, 45, 40], "score": 0.91, "class_id": 1, "class_name": "firstobject"},
        {"box": [55, 3, 75, 33], "score": 0.89, "class_id": 0, "class_name": "hand"},
        {"box": [40, 35, 80, 70], "score": 0.86, "class_id": 2, "class_name": "secondobject"},
        {"box": [82, 20, 99, 50], "score": 0.62, "class_id": 1, "class_name": "firstobject"},
    ]
    hf = [
        {"a": 0, "b": 1, "prob": 0.88},
        {"a": 2, "b": 1, "prob": 0.79},
    ]
    fs = [
        {"a": 1, "b": 3, "prob": 0.94},
        {"a": 4, "b": 3, "prob": 0.93},
    ]
    return {
        "type": "video",
        "video_path": "/upstream/demo/example.mp4",
        "fps": 30.0,
        "width": 100,
        "height": 80,
        "num_frames": 2,
        "score_thr": 0.3,
        "nms_iou": 0.5,
        "frame_stride": 2,
        "class_names": ["hand", "firstobject", "secondobject"],
        "frames": [
            {
                "frame_idx": 0,
                "processed": True,
                "detections": copy.deepcopy(detections),
                "hf": copy.deepcopy(hf),
                "fs": copy.deepcopy(fs),
            },
            {
                "frame_idx": 1,
                "processed": False,
                "detections": [],
                "hf": [],
                "fs": [],
            },
        ],
    }


def _normalize(payload=None):
    return normalize_official_video(
        _official_payload() if payload is None else payload,
        dataset="hoi4d",
        video_id="clip__001",
        video_path="D:/dataset/clip/image.mp4",
        source_metadata={
            "checkpoint": "epoch_5.pth",
            "config": "co_dino_hoi.py",
            "exporter": {"commit": "abc123"},
        },
    )


class NormalizeOfficialVideoTests(unittest.TestCase):
    def _assert_invalid(self, mutate, message_fragment=None):
        payload = _official_payload()
        mutate(payload)
        with self.assertRaises(CandidateValidationError) as raised:
            _normalize(payload)
        if message_fragment is not None:
            self.assertIn(message_fragment, str(raised.exception))

    def test_normalizes_identity_detections_and_all_index_links(self):
        candidates = _normalize()

        self.assertEqual(candidates["schema_version"], SCHEMA_VERSION)
        self.assertEqual(candidates["dataset"], "hoi4d")
        self.assertEqual(candidates["video_id"], "clip__001")
        self.assertEqual(candidates["video"]["path"], "D:/dataset/clip/image.mp4")
        self.assertEqual(candidates["source"]["official_video_path"], "/upstream/demo/example.mp4")
        self.assertEqual(candidates["source"]["metadata"]["checkpoint"], "epoch_5.pth")

        frame = candidates["frames"][0]
        self.assertEqual(len(frame["detections"]), 5)
        self.assertEqual(len(frame["links"]["hf"]), 2)
        self.assertEqual(len(frame["links"]["fs"]), 2)
        self.assertEqual(
            [link["source_detection_index"] for link in frame["links"]["hf"]],
            [0, 2],
        )
        self.assertEqual(
            [link["target_detection_index"] for link in frame["links"]["hf"]],
            [1, 1],
        )
        self.assertEqual(frame["links"]["hf"][1]["source_detection_id"], "f000000_d0002")
        self.assertEqual(frame["links"]["fs"][1]["source_detection_id"], "f000000_d0004")
        validate_candidates(candidates)

    def test_rejects_frame_count_and_non_continuous_frame_index(self):
        self._assert_invalid(
            lambda payload: payload.__setitem__("num_frames", 3),
            "does not match num_frames",
        )
        self._assert_invalid(
            lambda payload: payload["frames"][1].__setitem__("frame_idx", 2),
            "continuous absolute frame index",
        )
        self._assert_invalid(
            lambda payload: payload["frames"][0].__setitem__("processed", 1),
            "expected a boolean",
        )

    def test_rejects_invalid_class_id_or_name(self):
        self._assert_invalid(
            lambda payload: payload["frames"][0]["detections"][0].__setitem__("class_id", 3),
            "must be <= 2",
        )
        self._assert_invalid(
            lambda payload: payload["frames"][0]["detections"][0].__setitem__("class_id", True),
            "expected an integer",
        )
        self._assert_invalid(
            lambda payload: payload["frames"][0]["detections"][0].__setitem__("class_name", "firstobject"),
            "requires 'hand'",
        )

    def test_rejects_non_finite_unordered_or_out_of_bounds_xyxy(self):
        invalid_boxes = (
            ([1, 2, math.nan, 10], "finite"),
            ([5, 2, 5, 10], "x1 < x2"),
            ([1, 9, 10, 9], "y1 < y2"),
            ([-1, 2, 10, 20], "within [0, 100]"),
            ([1, 2, 101, 20], "within [0, 100]"),
            ([1, 2, 10, 81], "within [0, 80]"),
            ([1, 2, 10], "four xyxy"),
        )
        for box, message in invalid_boxes:
            with self.subTest(box=box):
                self._assert_invalid(
                    lambda payload, value=box: payload["frames"][0]["detections"][0].__setitem__("box", value),
                    message,
                )

    def test_rejects_invalid_detection_score(self):
        for score in (-0.01, 1.01, math.inf, True):
            with self.subTest(score=score):
                self._assert_invalid(
                    lambda payload, value=score: payload["frames"][0]["detections"][0].__setitem__("score", value)
                )

    def test_rejects_out_of_range_or_wrong_type_link_endpoints(self):
        self._assert_invalid(
            lambda payload: payload["frames"][0]["hf"][0].__setitem__("a", 1),
            "hf source must be class 0",
        )
        self._assert_invalid(
            lambda payload: payload["frames"][0]["hf"][0].__setitem__("b", 3),
            "hf target must be class 1",
        )
        self._assert_invalid(
            lambda payload: payload["frames"][0]["fs"][0].__setitem__("a", 0),
            "fs source must be class 1",
        )
        self._assert_invalid(
            lambda payload: payload["frames"][0]["fs"][0].__setitem__("b", 1),
            "fs target must be class 2",
        )
        self._assert_invalid(
            lambda payload: payload["frames"][0]["hf"][0].__setitem__("a", 99),
            "out of range",
        )
        self._assert_invalid(
            lambda payload: payload["frames"][0]["hf"][0].__setitem__("a", True),
            "expected an integer",
        )

    def test_rejects_invalid_link_probability(self):
        for probability in (-0.01, 1.01, math.nan, False):
            with self.subTest(probability=probability):
                self._assert_invalid(
                    lambda payload, value=probability: payload["frames"][0]["hf"][0].__setitem__("prob", value)
                )

    def test_rejects_non_json_source_metadata(self):
        with self.assertRaises(CandidateValidationError):
            normalize_official_video(
                _official_payload(),
                dataset="hoi4d",
                video_id="clip",
                video_path="clip.mp4",
                source_metadata={"bad": math.nan},
            )


class CandidateDocumentTests(unittest.TestCase):
    def test_summary_and_report_count_all_candidates_without_collapsing(self):
        candidates = _normalize()
        summary = summarize_candidates(candidates)

        self.assertEqual(summary["num_frames"], 2)
        self.assertEqual(summary["num_processed_frames"], 1)
        self.assertEqual(summary["num_unprocessed_frames"], 1)
        self.assertEqual(summary["num_detections"], 5)
        self.assertEqual(
            summary["detections_by_class"],
            {"hand": 2, "firstobject": 2, "secondobject": 1},
        )
        self.assertEqual(summary["num_links"], 4)
        self.assertEqual(summary["links_by_kind"], {"hf": 2, "fs": 2})
        self.assertEqual(summary["processed_frames_only"]["num_detections"], 5)
        self.assertEqual(summary["processed_frames_only"]["num_links"], 4)

        report = build_report(candidates)
        self.assertEqual(report["schema_version"], REPORT_SCHEMA_VERSION)
        self.assertEqual(report["candidate_schema_version"], SCHEMA_VERSION)
        self.assertEqual(report["dataset"], "hoi4d")
        self.assertEqual(report["video_id"], "clip__001")
        self.assertEqual(len(report["frames"]), 2)
        self.assertEqual(report["frames"][0]["links_by_kind"], {"hf": 2, "fs": 2})

    def test_json_string_round_trip_is_lossless(self):
        candidates = _normalize()
        encoded = dumps_candidates(candidates)
        decoded = loads_candidates(encoded)
        self.assertEqual(decoded, candidates)

    def test_file_round_trip_and_official_file_loader(self):
        candidates = _normalize()
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            candidates_path = root / "nested" / "candidates.json"
            report_path = root / "nested" / "report.json"
            official_path = root / "official.json"

            write_candidates_json(candidates, candidates_path)
            self.assertEqual(read_candidates_json(candidates_path), candidates)

            write_report_json(candidates, report_path)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report, build_report(candidates))

            write_json_atomic(official_path, _official_payload())
            from_official_file = read_official_video_json(
                official_path,
                dataset="hoi4d",
                video_id="clip__001",
                video_path="D:/dataset/clip/image.mp4",
                source_metadata={
                    "checkpoint": "epoch_5.pth",
                    "config": "co_dino_hoi.py",
                    "exporter": {"commit": "abc123"},
                },
            )
            self.assertEqual(from_official_file, candidates)

    def test_atomic_writer_replaces_document_and_rejects_nan(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "payload.json"
            self.assertIsNone(write_json_atomic(path, {"version": 1}))
            self.assertIsNone(write_json_atomic(path, {"version": 2}))
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"version": 2})

            with self.assertRaises(CandidateValidationError):
                write_json_atomic(path, {"bad": math.nan})
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"version": 2})
            self.assertEqual(list(path.parent.glob(".payload.json.*.tmp")), [])

    def test_normalized_validator_detects_id_and_endpoint_tampering(self):
        candidates = _normalize()
        bad_detection_id = copy.deepcopy(candidates)
        bad_detection_id["frames"][0]["detections"][0]["detection_id"] = "wrong"
        with self.assertRaises(CandidateValidationError):
            validate_candidates(bad_detection_id)

        bad_endpoint_id = copy.deepcopy(candidates)
        bad_endpoint_id["frames"][0]["links"]["hf"][0]["target_detection_id"] = "wrong"
        with self.assertRaises(CandidateValidationError):
            validate_candidates(bad_endpoint_id)

    def test_json_loader_rejects_non_finite_constants(self):
        encoded = dumps_candidates(_normalize())
        corrupted = encoded.replace('"score": 0.97', '"score": NaN', 1)
        with self.assertRaises(CandidateValidationError):
            loads_candidates(corrupted)


if __name__ == "__main__":
    unittest.main()
