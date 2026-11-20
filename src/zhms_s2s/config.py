from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .errors import ModelUnavailableError


LANGUAGES = ("zh", "ms")


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(slots=True)
class Settings:
    project_root: Path = field(default_factory=lambda: Path.cwd())
    offline: bool = True
    device: str = "cpu"
    models_dir: Path = Path("models")
    outputs_dir: Path = Path("outputs")
    asr_backend: str = "mlx-whisper"
    asr_ms_backend: str = "qwen-context"
    asr_model: Path = Path("models/malaysian-whisper-small-v3")
    asr_ms_model: Path = Path("models/malaysian-whisper-small-v3")
    asr_mlx_model: Path = Path(
        "models/mlx-malaysian-whisper-small-v3-f16-standard-vocab"
    )
    asr_qwen_model: Path = Path("models/qwen3-asr-1.7b-4bit")
    lid_model: Path = Path("models/lid-voxlingua107-coreml")
    asr_num_beams: int = 1
    asr_ms_device: str = "cpu"
    asr_ms_num_beams: int = 1
    asr_ms_segment_num_beams: int = 1
    asr_ms_split_on_pause: bool = False
    asr_ms_min_silence_ms: int = 380
    asr_ms_min_segment_s: float = 1.2
    asr_ms_max_segments: int = 4
    asr_prompts: dict[str, str] = field(default_factory=dict)
    asr_preprocess_audio: bool = True
    # Emergency-stable mode: keep the enhancement implementation available,
    # but bypass it in the default ASR -> MT -> TTS path.
    baseline_direct_pipeline: bool = True
    entity_enhancement_enabled: bool = False
    entity_prompt_enabled: bool = False
    entity_prompt_max_terms: int = 24
    entity_prompt_max_characters: int = 280
    entity_lexicon: Path = Path("data/entity_lexicon.json")
    # A transcript-retrieved, request-scoped entity context. This is separate
    # from the rejected global lexicon prompt and is safe to use in direct mode.
    selective_entity_rescue_enabled: bool = False
    selective_entity_rescue_max_candidates: int = 3
    selective_entity_rescue_max_prompt_characters: int = 96
    mt_backend: str = "hy-mt-llama"
    mt_num_beams: int = 1
    mt_direct_model: Path = Path("models/nllb-200-distilled-600M")
    mt_ct2_model: Path = Path("models/nllb-200-distilled-600M-ct2-int8")
    mt_ct2_threads: int = 4
    mt_hy_model: Path = Path(
        "models/hy-mt1.5-1.8b-q4/HY-MT1.5-1.8B-Q4_K_M.gguf"
    )
    mt_llama_server_path: Path = Path("/opt/homebrew/bin/llama-server")
    mt_hy_port: int = 18081
    mt_hy_gpu_layers: int = 99
    mt_terminology_enabled: bool = False
    mt_critical_slot_protection_enabled: bool = False
    mt_zh_en_model: Path = Path("models/argos/translate-zh_en-1_9")
    mt_en_ms_model: Path = Path("models/argos/translate-en_ms-1_9")
    mt_ms_en_model: Path = Path("models/argos/translate-ms_en-1_9")
    mt_en_zh_model: Path = Path("models/argos/translate-en_zh-1_9")
    tts_backend: str = "macos-say"
    tts_piper_model: Path = Path("models/piper/zh_CN-huayan-medium/zh_CN-huayan-medium.onnx")
    tts_piper_config: Path = Path("models/piper/zh_CN-huayan-medium/zh_CN-huayan-medium.onnx.json")
    tts_piper_alt_model: Path = Path(
        "models/piper/zh_CN-xiao_ya-medium/zh_CN-xiao_ya-medium.onnx"
    )
    tts_piper_alt_config: Path = Path(
        "models/piper/zh_CN-xiao_ya-medium/zh_CN-xiao_ya-medium.onnx.json"
    )
    tts_default_zh_voice: str = "xiao_ya"
    tts_piper_data_dir: Path = Path("models/piper/resources")
    tts_piper_tokenizer_dir: Path = Path(
        "models/piper/resources/bert-base-chinese"
    )
    tts_mms_model: Path = Path("models/mms-tts-zlm")
    tts_voices: dict[str, str] = field(
        default_factory=lambda: {"zh": "Tingting", "ms": "Amira"}
    )
    max_upload_mb: int = 25
    max_audio_seconds: float = 300.0
    keep_intermediate: bool = True

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Settings":
        config_path = Path(path or os.getenv("ZHMS_CONFIG", "config.json"))
        raw: dict[str, Any] = {}
        if config_path.exists():
            raw = json.loads(config_path.read_text(encoding="utf-8"))

        root = config_path.resolve().parent if config_path.exists() else Path.cwd().resolve()
        settings = cls(project_root=root)
        for key, value in raw.items():
            if not hasattr(settings, key) or key == "project_root":
                continue
            if key.endswith("_dir") or key.endswith("_model"):
                path_value = Path(value)
                value = path_value if path_value.is_absolute() else root / path_value
            setattr(settings, key, value)

        settings.offline = _bool_env("ZHMS_OFFLINE", settings.offline)
        settings.device = os.getenv("ZHMS_DEVICE", settings.device)
        for field_name in (
            "models_dir",
            "outputs_dir",
            "asr_model",
            "asr_ms_model",
            "asr_mlx_model",
            "asr_qwen_model",
            "lid_model",
            "entity_lexicon",
            "mt_direct_model",
            "mt_ct2_model",
            "mt_hy_model",
            "mt_llama_server_path",
            "mt_zh_en_model",
            "mt_en_ms_model",
            "mt_ms_en_model",
            "mt_en_zh_model",
            "tts_piper_model",
            "tts_piper_config",
            "tts_piper_alt_model",
            "tts_piper_alt_config",
            "tts_piper_data_dir",
            "tts_piper_tokenizer_dir",
            "tts_mms_model",
        ):
            value = Path(getattr(settings, field_name))
            if not value.is_absolute():
                value = root / value
            setattr(settings, field_name, value.resolve())
        settings.outputs_dir.mkdir(parents=True, exist_ok=True)
        return settings

    def enable_offline_environment(self) -> None:
        if not self.offline:
            return
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    def validate_model_paths(self) -> None:
        checks = {
            self.lid_model: (
                "config.json",
                "labels.json",
                "SpeechBrainECAPAVoxLingua107.mlmodelc/model.mil",
            ),
        }
        if self.asr_backend == "transformers-whisper":
            checks[self.asr_model] = ("config.json", "model.safetensors")
        elif self.asr_backend == "mlx-whisper":
            checks[self.asr_mlx_model] = ("config.json", "weights.safetensors")
        elif self.asr_backend == "qwen-context":
            checks[self.asr_qwen_model] = ("config.json", "weights.safetensors")
        else:
            raise ModelUnavailableError(
                f"Unsupported ASR backend '{self.asr_backend}'."
            )
        if self.asr_ms_backend == "transformers-whisper":
            checks[self.asr_ms_model] = ("config.json", "model.safetensors")
        elif self.asr_ms_backend == "mlx-whisper":
            checks[self.asr_mlx_model] = ("config.json", "weights.safetensors")
        elif self.asr_ms_backend == "qwen-context":
            checks[self.asr_qwen_model] = ("config.json", "weights.safetensors")
        else:
            raise ModelUnavailableError(
                f"Unsupported Malay ASR backend '{self.asr_ms_backend}'."
            )
        if self.mt_backend in {"nllb-direct", "nllb-ct2"}:
            checks[self.mt_direct_model] = ("config.json",)
        elif self.mt_backend == "argos-pivot":
            checks.update(
                {
                    self.mt_zh_en_model: ("model/model.bin", "sentencepiece.model"),
                    self.mt_en_ms_model: ("model/model.bin", "sentencepiece.model"),
                    self.mt_ms_en_model: ("model/model.bin", "sentencepiece.model"),
                    self.mt_en_zh_model: ("model/model.bin", "sentencepiece.model"),
                }
            )
        elif self.mt_backend != "hy-mt-llama":
            raise ModelUnavailableError(
                f"Unsupported MT backend '{self.mt_backend}'."
            )
        missing = [
            str(path / required)
            for path, required_files in checks.items()
            for required in required_files
            if not (path / required).is_file()
        ]
        if (
            (
                self.selective_entity_rescue_enabled
                or (
                    not self.baseline_direct_pipeline
                    and (self.entity_enhancement_enabled or self.entity_prompt_enabled)
                )
            )
            and not self.entity_lexicon.is_file()
        ):
            missing.append(str(self.entity_lexicon))
        if self.mt_backend in {"nllb-direct", "nllb-ct2"} and not any(
            (self.mt_direct_model / name).is_file()
            for name in ("model.safetensors", "pytorch_model.bin")
        ):
            missing.append(str(self.mt_direct_model / "model weights"))
        if self.mt_backend == "nllb-ct2":
            for path in (
                self.mt_ct2_model / "model.bin",
                self.mt_ct2_model / "config.json",
                self.mt_ct2_model / "shared_vocabulary.json",
                self.mt_direct_model / "tokenizer.json",
            ):
                if not path.is_file():
                    missing.append(str(path))
        if self.mt_backend == "hy-mt-llama":
            for path in (self.mt_hy_model, self.mt_llama_server_path):
                if not path.is_file():
                    missing.append(str(path))
        if self.tts_backend == "piper-mms":
            for path in (
                self.tts_piper_model,
                self.tts_piper_config,
                self.tts_piper_alt_model,
                self.tts_piper_alt_config,
                self.tts_piper_data_dir / "g2pW" / "g2pw.onnx",
                self.tts_piper_tokenizer_dir / "vocab.txt",
                self.tts_mms_model / "config.json",
                self.tts_mms_model / "model.safetensors",
            ):
                if not path.is_file():
                    missing.append(str(path))
        if missing:
            raise ModelUnavailableError(
                "Required local model directories are missing; run scripts/prepare_models.py first.",
                detail={"missing": missing},
            )


def validate_language(language: str) -> str:
    normalized = language.strip().lower()
    aliases = {"zh-cn": "zh", "zh_cn": "zh", "cmn": "zh", "ms-my": "ms", "ms_my": "ms", "zlm": "ms"}
    normalized = aliases.get(normalized, normalized)
    if normalized not in LANGUAGES:
        from .errors import InvalidLanguageError

        raise InvalidLanguageError(
            f"Unsupported language '{language}'. Supported values: zh, ms."
        )
    return normalized


def target_for(source: str) -> str:
    source = validate_language(source)
    return "ms" if source == "zh" else "zh"
