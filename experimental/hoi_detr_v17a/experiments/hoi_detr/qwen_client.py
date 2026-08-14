"""Qwen (通义千问) OpenAI 兼容客户端。**默认走我们自部署的 vLLM，不依赖 openai 包。**

============================ 2026-08-14 改了什么、为什么 ============================

改动一：**默认端点从阿里云 MaaS 换成本机 vLLM**（`http://127.0.0.1:8807/v1`）。

  我们在 UCB 8 卡机 GPU7 上已经部署了 Qwen（`--served-model-name vlm --port 8807`），
  本地经 `tools/vlm_tunnel.sh` 开隧道即可访问。原默认指向阿里云公有云，有三个问题：
    * 要 `DASHSCOPE_API_KEY`，而那个 key **2026-08-11 泄漏进过 git 历史、至今未吊销重签**；
    * 把视频帧传到外部服务；
    * 与管线其余部分（`vlm_transparency_gate` 等）各走各的服务，结论无法互相印证。
  仍可用 `QWEN_BASE_URL` 指回阿里云（那时才需要 key），行为不变。

改动二：**去掉 `openai` 依赖，改用标准库 urllib**。

  实测 `biv2ap` 与 `codetr` 两个 env、本地与 UCB 两台机器，**四处都没有 `openai` 包**。
  而调用方 `scale_fusion.py` 捕获 ImportError 后会**静默降级为纯几何尺度** ——
  也就是尺度融合看起来跑过了、实际没生效，跑完 18 分钟才从 `scale_verdict` 发现。
  与其在四个 env 里各装一遍再维护版本，不如去掉这个依赖：本模块只需要
  "POST 一个 JSON、读回一个 JSON"，urllib 足够，且与 `ego_pipeline/bin/
  vlm_transparency_gate.py` 的做法一致（那边一直是 urllib）。

接口**完全不变**：`make_client` / `call_qwen(..., client=…)` / `build_user_content` /
`image_to_data_url` 签名与返回值都保持原样，四个调用方无需改动。

============================ 本地 vLLM 与 DashScope 的两处差异 ============================

(沿用原注释，这两条是踩出来的，别删)

 1) 关思考链的参数名不同 —— DashScope 用 `extra_body.enable_thinking`，
    vLLM 用 `chat_template_kwargs`；不设的话 Qwen3.5 会先长篇推理，早期被 max_tokens
    截断就永远等不到 JSON（实测：300 token 全用在推理上，一个花括号都没吐）。
 2) vLLM 支持 `response_format=json_object`（启动时已开 xgrammar 结构化输出），
    直接从解码层保证合法 JSON，比在 prompt 里求它"只输出 JSON"可靠得多。

⚠ 判断"是不是本地 vLLM"从**显式环境变量** `QWEN_LOCAL_VLLM=1` 改为**按端点自动判断**
  （非 aliyuncs 即视为本地 vLLM），并保留 `QWEN_LOCAL_VLLM=0` 强制关闭。原来忘了设那个
  变量就会退化成 DashScope 参数、在 vLLM 上关不掉思考链，是个只在运行时才暴露的坑。
"""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

# 与 ego_pipeline/bin/vlm_transparency_gate.py 同源: VLM_API_BASE / QWEN_BASE_URL 等价
BASE_URL = (os.getenv("QWEN_BASE_URL") or os.getenv("VLM_API_BASE")
            or "http://127.0.0.1:8807/v1")
MODEL = os.getenv("QWEN_MODEL", "vlm")          # vLLM 的 --served-model-name
API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
TIMEOUT = int(os.getenv("QWEN_TIMEOUT", "600"))

if not API_KEY and "aliyuncs.com" in BASE_URL:
    raise SystemExit(
        "qwen_client: 指向阿里云需要 DASHSCOPE_API_KEY(旧硬编码 key 已泄漏进 git 历史, "
        "2026-08-11 已从代码移除并需吊销重签)。\n"
        "  推荐改用自部署 vLLM: 不设 QWEN_BASE_URL 即默认 http://127.0.0.1:8807/v1, 无需 key。\n"
        "  本地需先开隧道: ./tools/vlm_tunnel.sh")


def _is_local_vllm(base_url: str) -> bool:
    forced = os.getenv("QWEN_LOCAL_VLLM")
    if forced is not None:
        return forced == "1"
    return "aliyuncs.com" not in base_url


@dataclass
class QwenClient:
    """轻量客户端。只保存端点与 key —— 请求用 urllib 现发, 无需 openai 包。"""
    base_url: str = BASE_URL
    api_key: str = API_KEY


def make_client() -> QwenClient:
    return QwenClient()


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
    client: QwenClient | None = None,
    model: str = MODEL,
    enable_thinking: bool = False,
    temperature: float | None = 0.0,
) -> QwenResponse:
    client = client or make_client()
    body: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "stream": False,
        "enable_thinking": enable_thinking,      # DashScope 口径
    }
    if temperature is not None:
        body["temperature"] = temperature
    if _is_local_vllm(client.base_url):
        body["chat_template_kwargs"] = {"enable_thinking": enable_thinking}
        body["response_format"] = {"type": "json_object"}
        body["max_tokens"] = int(os.getenv("QWEN_MAX_TOKENS", "1024"))

    headers = {"Content-Type": "application/json"}
    if client.api_key:
        headers["Authorization"] = f"Bearer {client.api_key}"
    req = urllib.request.Request(client.base_url.rstrip("/") + "/chat/completions",
                                 data=json.dumps(body).encode("utf-8"), headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            out = json.loads(resp.read())
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        raise RuntimeError(
            f"qwen_client: 服务不可达({client.base_url}): {exc}. "
            f"自部署 vLLM 在 UCB 8 卡机 GPU7, 本地需先 ./tools/vlm_tunnel.sh") from exc

    msg = (out.get("choices") or [{}])[0].get("message") or {}
    return QwenResponse(content=msg.get("content") or "",
                        reasoning=msg.get("reasoning_content") or "")
