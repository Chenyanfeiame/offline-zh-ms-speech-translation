"""Small translation worker that keeps CTranslate2 isolated from PyTorch/OpenMP."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from .backends.mt import ArgosPivotTranslator


def main() -> None:
    marker = "ZHMS_RESULT="
    try:
        payload = json.load(sys.stdin)
        paths = payload["paths"]
        translator = ArgosPivotTranslator(
            Path(paths["zh-en"]),
            Path(paths["en-ms"]),
            Path(paths["ms-en"]),
            Path(paths["en-zh"]),
            device=payload.get("device", "cpu"),
            offline=bool(payload.get("offline", True)),
            isolate_process=False,
        )
        translated, pivot = translator._translate_local(
            payload["text"],
            payload["source_language"],
            payload["target_language"],
        )
        result = {
            "ok": True,
            "translated_text": translated,
            "pivot_text": pivot,
        }
    except Exception as exc:
        result = {
            "ok": False,
            "error": {"type": type(exc).__name__, "reason": str(exc)},
        }
    print(marker + json.dumps(result, ensure_ascii=False, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
