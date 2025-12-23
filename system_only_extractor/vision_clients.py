from __future__ import annotations

import base64
import os
from dataclasses import dataclass
from typing import Literal, Optional


Provider = Literal["OPENAI", "ANTHROPIC"]


@dataclass
class VisionConfig:
    enabled: bool
    provider: Provider
    min_confidence: int
    openai_model: str
    anthropic_model: str


def load_vision_config() -> VisionConfig:
    enabled = os.getenv("VISION_ENABLED", "0").strip() in {"1", "true", "TRUE", "yes", "YES"}
    provider = os.getenv("VISION_PROVIDER", "OPENAI").strip().upper()
    if provider not in {"OPENAI", "ANTHROPIC"}:
        provider = "OPENAI"

    try:
        min_confidence = int(os.getenv("VISION_MIN_CONFIDENCE", "55"))
    except ValueError:
        min_confidence = 55

    return VisionConfig(
        enabled=enabled,
        provider=provider,  # type: ignore[assignment]
        min_confidence=min_confidence,
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        anthropic_model=os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022"),
    )


def _b64_png(png_bytes: bytes) -> str:
    return base64.b64encode(png_bytes).decode("ascii")


def ask_vision_for_sources(
    *,
    png_bytes: bytes,
    equipment_name: str,
    instruction: str,
    provider: Provider,
    model: str,
) -> str:
    if provider == "OPENAI":
        return _ask_openai(png_bytes, equipment_name, instruction, model=model)
    return _ask_anthropic(png_bytes, equipment_name, instruction, model=model)


def _ask_openai(png_bytes: bytes, equipment_name: str, instruction: str, model: str) -> str:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    from openai import OpenAI

    client = OpenAI(api_key=api_key)

    b64 = _b64_png(png_bytes)
    prompt = (
        "You are reading an electrical single-line diagram crop. "
        "Determine the upstream power source equipment feeding the given equipment. "
        "Primary = nearest top-most upstream source. Alternate = second upstream source if present. "
        "If no alternate, return '-' for alternate. "
        "Return STRICT JSON: {\"primary\": string, \"alternate\": string}.\n\n"
        f"Equipment: {equipment_name}\n"
        f"Extra instruction: {instruction}\n"
    )

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                ],
            }
        ],
        temperature=0,
    )
    return resp.choices[0].message.content or ""


def _ask_anthropic(png_bytes: bytes, equipment_name: str, instruction: str, model: str) -> str:
    api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set")

    from anthropic import Anthropic

    client = Anthropic(api_key=api_key)

    b64 = _b64_png(png_bytes)
    prompt = (
        "You are reading an electrical single-line diagram crop. "
        "Determine the upstream power source equipment feeding the given equipment. "
        "Primary = nearest top-most upstream source. Alternate = second upstream source if present. "
        "If no alternate, return '-' for alternate. "
        "Return STRICT JSON: {\"primary\": string, \"alternate\": string}.\n\n"
        f"Equipment: {equipment_name}\n"
        f"Extra instruction: {instruction}\n"
    )

    msg = client.messages.create(
        model=model,
        max_tokens=400,
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": b64,
                        },
                    },
                ],
            }
        ],
    )

    out_parts = []
    for part in msg.content:
        if getattr(part, "type", None) == "text":
            out_parts.append(part.text)
    return "\n".join(out_parts)
