"""Qwen (通义千问) OpenAI-compatible client used by the optional click-frame mode."""

from __future__ import annotations

import base64
import mimetypes
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from openai import OpenAI

BASE_URL = "https://ws-mn8oji6058okce6e.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
MODEL = os.getenv("QWEN_MODEL", "qwen3.8-max")
API_KEY = os.getenv(
    "DASHSCOPE_API_KEY",
    "sk-ws-H.EHDRRER.eCKD.MEUCIHCWcRPtx-zr47BtPDrVN-ERMZSyTvVoDbhHS9fqzmiiAiEA3Z8bf-JIDRJgIlW7aRhd-VFtsa2HgTrBN3BtwXKFycI",
)


def make_client() -> OpenAI:
    return OpenAI(api_key=API_KEY, base_url=BASE_URL)


def image_to_data_url(image_path: str | Path) -> str:
    image_path = Path(image_path)
    mime = mimetypes.guess_type(str(image_path))[0] or "image/png"
    b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def build_user_content(
    text: str, image_paths: Iterable[str | Path] | None = None
) -> str | list[dict[str, Any]]:
    image_paths = list(image_paths or [])
    if not image_paths:
        return text
    parts: list[dict[str, Any]] = [{"type": "text", "text": text}]
    for path in image_paths:
        parts.append({"type": "image_url", "image_url": {"url": image_to_data_url(path)}})
    return parts


@dataclass
class QwenResponse:
    content: str
    reasoning: str


def call_qwen(
    system_prompt: str,
    user_content: str | list[dict[str, Any]],
    *,
    client: OpenAI | None = None,
    model: str = MODEL,
    enable_thinking: bool = False,
    temperature: float | None = 0.0,
) -> QwenResponse:
    client = client or make_client()
    kwargs: dict[str, Any] = {"extra_body": {"enable_thinking": enable_thinking}}
    if temperature is not None:
        kwargs["temperature"] = temperature
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        stream=False,
        **kwargs,
    )
    message = completion.choices[0].message
    return QwenResponse(
        content=message.content or "",
        reasoning=getattr(message, "reasoning_content", None) or "",
    )
