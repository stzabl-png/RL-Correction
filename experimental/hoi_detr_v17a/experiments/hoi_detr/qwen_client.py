"""Qwen (通义千问) OpenAI-compatible client used by the optional click-frame mode."""

from __future__ import annotations

import base64
import mimetypes
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from openai import OpenAI

# 端点与模型可由环境变量覆盖, 默认仍是阿里云 MaaS —— 这样同一份代码既能走外部 API,
# 也能指向我们 8 卡机上自部署的 vLLM(OpenAI 兼容, --served-model-name vlm --port 8807)。
BASE_URL = os.getenv("QWEN_BASE_URL",
                     "https://ws-mn8oji6058okce6e.cn-beijing.maas.aliyuncs.com/compatible-mode/v1")
MODEL = os.getenv("QWEN_MODEL", "qwen3.7-plus")
API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
if not API_KEY and "aliyuncs.com" in BASE_URL:
    raise SystemExit("qwen_client: 设置 DASHSCOPE_API_KEY 环境变量(旧硬编码 key 已泄漏进 git 历史, "
                     "2026-08-11 已从代码移除并需吊销重签; 本地 vLLM 走 QWEN_BASE_URL 不需要 key)")


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
    # 指向自部署 vLLM 时开 QWEN_LOCAL_VLLM=1。两处差异必须显式处理, 否则输出解析不了:
    #  1) 关思考链的参数名不同 —— DashScope 用 extra_body.enable_thinking,
    #     vLLM 用 chat_template_kwargs;不设的话 Qwen3.5 会先长篇推理, 早期被 max_tokens 截断
    #     就永远等不到 JSON(实测:300 token 全用在推理上, 一个花括号都没吐)。
    #  2) vLLM 支持 response_format=json_object(启动时已开 xgrammar 结构化输出),
    #     直接从解码层保证合法 JSON, 比在 prompt 里求它"只输出JSON"可靠得多。
    if os.getenv("QWEN_LOCAL_VLLM") == "1":
        kwargs["extra_body"]["chat_template_kwargs"] = {"enable_thinking": enable_thinking}
        kwargs["response_format"] = {"type": "json_object"}
        kwargs["max_tokens"] = int(os.getenv("QWEN_MAX_TOKENS", "1024"))
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
