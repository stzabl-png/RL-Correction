"""Repository-owned helpers for the standalone HOI-DETR experiment."""

from .adapter import (
    CLASS_NAMES,
    REPORT_SCHEMA_VERSION,
    SCHEMA_VERSION,
    CandidateValidationError,
    build_report,
    dumps_candidates,
    loads_candidates,
    normalize_official_video,
    normalize_official_video_json,
    read_candidates_json,
    read_official_video_json,
    summarize_candidates,
    validate_candidates,
    write_candidates_json,
    write_json_atomic,
    write_report_json,
)

__all__ = [
    "CLASS_NAMES",
    "REPORT_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "CandidateValidationError",
    "build_report",
    "dumps_candidates",
    "loads_candidates",
    "normalize_official_video",
    "normalize_official_video_json",
    "read_candidates_json",
    "read_official_video_json",
    "summarize_candidates",
    "validate_candidates",
    "write_candidates_json",
    "write_json_atomic",
    "write_report_json",
]
