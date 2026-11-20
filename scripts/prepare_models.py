#!/usr/bin/env python3
"""Download only the pinned models used by the final offline demo."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tarfile
import time
import urllib.request
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")

from huggingface_hub import snapshot_download


HF_MODELS = {
    "qwen3-asr-1.7b-4bit": {
        "repo_id": "moona3k/mlx-qwen3-asr-1.7b-4bit",
        "revision": "b2315a2537123353af98e37fc93eae078f1b7cf3",
        "license": "Apache-2.0",
        "purpose": "Shared Chinese/Malay ASR",
        "allow_patterns": ["*.json", "*.txt", "*.safetensors", "README.md"],
        "ready_files": ["config.json", "weights.safetensors"],
        "weight_file": "weights.safetensors",
    },
    "lid-voxlingua107-coreml": {
        "repo_id": "aufklarer/SpeechBrain-ECAPA-VoxLingua107-21M-CoreML",
        "revision": "2aa4d715a79e410d5f9aa32bd7a4fc9225bf9eb0",
        "license": "Apache-2.0",
        "purpose": "Apple-native open-set language identification",
        "allow_patterns": [
            "config.json",
            "labels.json",
            "artifact_manifest.json",
            "validation.json",
            "LICENSE",
            "README.md",
            "SpeechBrainECAPAVoxLingua107.mlmodelc/model.mil",
            "SpeechBrainECAPAVoxLingua107.mlmodelc/coremldata.bin",
            "SpeechBrainECAPAVoxLingua107.mlmodelc/weights/weight.bin",
            "SpeechBrainECAPAVoxLingua107.mlmodelc/analytics/coremldata.bin",
        ],
        "ready_files": [
            "config.json",
            "labels.json",
            "SpeechBrainECAPAVoxLingua107.mlmodelc/model.mil",
            "SpeechBrainECAPAVoxLingua107.mlmodelc/weights/weight.bin",
        ],
        "weight_file": "SpeechBrainECAPAVoxLingua107.mlmodelc/weights/weight.bin",
    },
    "hy-mt1.5-1.8b-q4": {
        "repo_id": "tencent/HY-MT1.5-1.8B-GGUF",
        "revision": "265b2e615a7dc9b06c435dc878829ad99a512ba2",
        "license": "See upstream model card",
        "purpose": "Direct Chinese/Malay translation via llama.cpp",
        "allow_patterns": ["HY-MT1.5-1.8B-Q4_K_M.gguf", "README.md"],
        "ready_files": ["HY-MT1.5-1.8B-Q4_K_M.gguf"],
        "weight_file": "HY-MT1.5-1.8B-Q4_K_M.gguf",
    },
    "mms-tts-zlm": {
        "repo_id": "facebook/mms-tts-zlm",
        "revision": "5a6309e9f695f23e922cf12595e229259ec827d4",
        "license": "CC-BY-NC-4.0",
        "purpose": "Malay TTS",
        "allow_patterns": ["*.json", "*.safetensors", "README.md"],
        "ready_files": ["config.json", "model.safetensors"],
        "weight_file": "model.safetensors",
    },
}

PIPER_VOICES = {
    "zh_CN-huayan-medium": {
        "revision": "04ca88f5685ca5253c701af3687e155b113bd512",
        "remote_path": "zh/zh_CN/huayan/medium",
        "files": {
            "zh_CN-huayan-medium.onnx": 63_201_294,
            "zh_CN-huayan-medium.onnx.json": 4_822,
            "MODEL_CARD": 276,
        },
    },
    "zh_CN-xiao_ya-medium": {
        "revision": "6428fc718959cf446b00df24966c0185b7f7fcbe",
        "remote_path": "zh/zh_CN/xiao_ya/medium",
        "files": {
            "zh_CN-xiao_ya-medium.onnx": 63_221_984,
            "zh_CN-xiao_ya-medium.onnx.json": 2_927,
            "MODEL_CARD": 434,
        },
    },
}

G2PW = {
    "url": (
        "https://huggingface.co/datasets/rhasspy/piper-checkpoints/resolve/"
        "0d8f505885b1fd426191b9d09d9b38f4e11408a0/zh/zh_CN/_resources/g2pw.tar.gz"
    ),
    "bytes": 112_991_734,
    "sha256": "252c6f561c11014fbb243d70aec993a816d82109887b91edcda0261ab9967060",
}

BERT_TOKENIZER = {
    "repo_id": "google-bert/bert-base-chinese",
    "revision": "aa70e8aa0516b4cc86ac6c1ec25c03e4be7e6ad8",
    "allow_patterns": ["config.json", "tokenizer.json", "tokenizer_config.json", "vocab.txt"],
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_resumable(url: str, destination: Path, expected_bytes: int) -> None:
    if destination.is_file() and destination.stat().st_size == expected_bytes:
        print(f"Reusing {destination}", flush=True)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    for attempt in range(1, 6):
        offset = partial.stat().st_size if partial.exists() else 0
        request = urllib.request.Request(
            url,
            headers={"Range": f"bytes={offset}-", "User-Agent": "zhms-offline-demo/0.1"},
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                mode = "ab" if offset and response.status == 206 else "wb"
                with partial.open(mode) as handle:
                    while chunk := response.read(1024 * 1024):
                        handle.write(chunk)
            if partial.stat().st_size != expected_bytes:
                raise OSError(f"Expected {expected_bytes} bytes, got {partial.stat().st_size}")
            partial.replace(destination)
            return
        except Exception:
            if attempt == 5:
                raise
            time.sleep(attempt * 2)


def safe_extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as bundle:
        root = destination.resolve()
        for member in bundle.getmembers():
            resolved = (destination / member.name).resolve()
            if resolved != root and root not in resolved.parents:
                raise RuntimeError(f"Unsafe path in archive: {member.name}")
        bundle.extractall(destination)


def add_manifest_entry(
    manifest: dict,
    name: str,
    weight: Path,
    models_dir: Path,
    **metadata: object,
) -> None:
    manifest["models"].append(
        {
            "name": name,
            **metadata,
            "weight_file": str(weight.relative_to(models_dir)),
            "weight_bytes": weight.stat().st_size,
            "weight_sha256": sha256(weight),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    args.models_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, list[dict]] = {"models": []}

    for name, spec in HF_MODELS.items():
        target = args.models_dir / name
        ready = all((target / item).is_file() for item in spec["ready_files"])
        if args.force or not ready:
            print(f"Downloading {spec['repo_id']} -> {target}", flush=True)
            snapshot_download(
                repo_id=spec["repo_id"],
                revision=spec["revision"],
                local_dir=target,
                allow_patterns=spec["allow_patterns"],
                max_workers=2,
            )
        weight = target / spec["weight_file"]
        add_manifest_entry(
            manifest,
            name,
            weight,
            args.models_dir,
            repo_id=spec["repo_id"],
            revision=spec["revision"],
            license=spec["license"],
            purpose=spec["purpose"],
        )

    endpoint = os.getenv("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    for voice_name, spec in PIPER_VOICES.items():
        root = args.models_dir / "piper" / voice_name
        base = f"{endpoint}/rhasspy/piper-voices/resolve/{spec['revision']}/{spec['remote_path']}"
        for filename, expected_bytes in spec["files"].items():
            download_resumable(f"{base}/{filename}", root / filename, expected_bytes)
        weight = root / f"{voice_name}.onnx"
        add_manifest_entry(
            manifest,
            f"piper-{voice_name}",
            weight,
            args.models_dir,
            repo_id="rhasspy/piper-voices",
            revision=spec["revision"],
            license="MIT",
            purpose="Mandarin TTS",
        )

    archive = args.models_dir / "piper" / "resources" / "g2pw.tar.gz"
    download_resumable(G2PW["url"], archive, G2PW["bytes"])
    if sha256(archive) != G2PW["sha256"]:
        raise RuntimeError("Piper g2pW resource SHA-256 mismatch")
    g2pw_root = args.models_dir / "piper" / "resources" / "g2pW"
    if args.force or not (g2pw_root / "g2pw.onnx").is_file():
        safe_extract(archive, g2pw_root)
    add_manifest_entry(
        manifest,
        "piper-g2pW-resource",
        g2pw_root / "g2pw.onnx",
        args.models_dir,
        source_url=G2PW["url"],
        license="MIT",
        purpose="Chinese polyphone phonemization",
    )

    tokenizer_root = args.models_dir / "piper" / "resources" / "bert-base-chinese"
    if args.force or not (tokenizer_root / "vocab.txt").is_file():
        snapshot_download(
            repo_id=BERT_TOKENIZER["repo_id"],
            revision=BERT_TOKENIZER["revision"],
            local_dir=tokenizer_root,
            allow_patterns=BERT_TOKENIZER["allow_patterns"],
            max_workers=2,
        )
    add_manifest_entry(
        manifest,
        "piper-bert-base-chinese-tokenizer",
        tokenizer_root / "vocab.txt",
        args.models_dir,
        repo_id=BERT_TOKENIZER["repo_id"],
        revision=BERT_TOKENIZER["revision"],
        license="Apache-2.0",
        purpose="Offline tokenizer for Piper g2pW",
    )

    manifest_path = args.models_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
