#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import socket
from pathlib import Path


def block_network() -> None:
    def denied(*_args, **_kwargs):
        raise RuntimeError("Network access blocked by scripts/offline_verify.py")

    socket.create_connection = denied
    original_socket = socket.socket

    class OfflineSocket(original_socket):
        def connect(self, *_args, **_kwargs):
            raise RuntimeError("Network access blocked by scripts/offline_verify.py")

        def connect_ex(self, *_args, **_kwargs):
            return 101

    socket.socket = OfflineSocket


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Auto-detect and run both pipeline directions with sockets disabled"
    )
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/smoke/manifest.jsonl")
    )
    parser.add_argument("--report", type=Path, default=Path("reports/offline_verification.json"))
    args = parser.parse_args()
    os.environ["ZHMS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    block_network()

    from zhms_s2s.config import Settings
    from zhms_s2s.runtime import build_pipeline

    pipeline = build_pipeline(Settings.load())
    rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line]
    results = []
    for row in rows:
        path = Path(row["audio_path"])
        result = pipeline.run(
            path.read_bytes(),
            filename=path.name,
            source_language=None,
        )
        if result.source_language != row["language"]:
            raise RuntimeError(
                f"Language ID mismatch for {path}: "
                f"expected {row['language']}, got {result.source_language}"
            )
        results.append(result.model_dump())
    report = {
        "strict_socket_block": True,
        "environment": {
            "ZHMS_OFFLINE": os.environ["ZHMS_OFFLINE"],
            "HF_HUB_OFFLINE": os.environ["HF_HUB_OFFLINE"],
            "TRANSFORMERS_OFFLINE": os.environ["TRANSFORMERS_OFFLINE"],
        },
        "directions_verified": sorted(
            f"{item['source_language']}->{item['target_language']}" for item in results
        ),
        "automatic_language_detection": True,
        "results": results,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
