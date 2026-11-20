from __future__ import annotations

from .backends import (
    ArgosPivotTranslator,
    CoreMLLanguageIdentifier,
    MacOSSayTTS,
    MlxWhisperASR,
    HyMtLlamaTranslator,
    NllbCTranslate2Translator,
    NllbDirectTranslator,
    PauseSegmentedASR,
    PiperMmsTTS,
    Qwen3MlxASR,
    RoutedWhisperASR,
    TransformersWhisperASR,
)
from .config import Settings
from .errors import ModelUnavailableError
from .pipeline import SpeechTranslationPipeline


def build_pipeline(settings: Settings | None = None) -> SpeechTranslationPipeline:
    settings = settings or Settings.load()
    settings.enable_offline_environment()
    if settings.asr_backend == "transformers-whisper":
        zh_asr = TransformersWhisperASR(
            settings.asr_model,
            device=settings.device,
            offline=settings.offline,
            num_beams=settings.asr_num_beams,
            prompts=settings.asr_prompts,
        )
    elif settings.asr_backend == "mlx-whisper":
        zh_asr = MlxWhisperASR(settings.asr_mlx_model, offline=settings.offline)
    elif settings.asr_backend == "qwen-context":
        zh_asr = Qwen3MlxASR(settings.asr_qwen_model)
    else:
        raise ModelUnavailableError(
            f"Unsupported ASR backend '{settings.asr_backend}'."
        )

    if settings.asr_ms_backend == "transformers-whisper":
        ms_core = TransformersWhisperASR(
            settings.asr_ms_model,
            device=settings.asr_ms_device,
            offline=settings.offline,
            num_beams=settings.asr_ms_num_beams,
            prompts=settings.asr_prompts,
        )
    elif settings.asr_ms_backend == "mlx-whisper":
        ms_core = (
            zh_asr
            if settings.asr_backend == "mlx-whisper"
            else MlxWhisperASR(settings.asr_mlx_model, offline=settings.offline)
        )
    elif settings.asr_ms_backend == "qwen-context":
        ms_core = (
            zh_asr
            if settings.asr_backend == "qwen-context"
            else Qwen3MlxASR(settings.asr_qwen_model)
        )
    else:
        raise ModelUnavailableError(
            f"Unsupported Malay ASR backend '{settings.asr_ms_backend}'."
        )
    ms_asr = (
        PauseSegmentedASR(
            ms_core,
            language="ms",
            min_silence_ms=settings.asr_ms_min_silence_ms,
            min_segment_s=settings.asr_ms_min_segment_s,
            max_segments=settings.asr_ms_max_segments,
            segmented_num_beams=settings.asr_ms_segment_num_beams,
        )
        if settings.asr_ms_split_on_pause
        else ms_core
    )
    asr = RoutedWhisperASR(
        {"zh": zh_asr, "ms": ms_asr},
        CoreMLLanguageIdentifier(settings.lid_model),
    )
    if settings.mt_backend == "nllb-direct":
        translator = NllbDirectTranslator(
            settings.mt_direct_model,
            device=settings.device,
            offline=settings.offline,
            num_beams=settings.mt_num_beams,
        )
    elif settings.mt_backend == "nllb-ct2":
        translator = NllbCTranslate2Translator(
            settings.mt_ct2_model,
            settings.mt_direct_model,
            offline=settings.offline,
            num_beams=settings.mt_num_beams,
            intra_threads=settings.mt_ct2_threads,
        )
    elif settings.mt_backend == "hy-mt-llama":
        translator = HyMtLlamaTranslator(
            settings.mt_hy_model,
            settings.mt_llama_server_path,
            port=settings.mt_hy_port,
            gpu_layers=settings.mt_hy_gpu_layers,
            terminology_enabled=settings.mt_terminology_enabled,
            critical_slot_protection_enabled=(
                settings.mt_critical_slot_protection_enabled
            ),
        )
    elif settings.mt_backend == "argos-pivot":
        translator = ArgosPivotTranslator(
            settings.mt_zh_en_model,
            settings.mt_en_ms_model,
            settings.mt_ms_en_model,
            settings.mt_en_zh_model,
            device=settings.device,
            offline=settings.offline,
        )
    else:
        raise ModelUnavailableError(f"Unsupported MT backend '{settings.mt_backend}'.")
    if settings.tts_backend == "macos-say":
        tts = MacOSSayTTS(settings.tts_voices)
    elif settings.tts_backend == "piper-mms":
        tts = PiperMmsTTS(
            {
                "huayan": (
                    settings.tts_piper_model,
                    settings.tts_piper_config,
                ),
                "xiao_ya": (
                    settings.tts_piper_alt_model,
                    settings.tts_piper_alt_config,
                ),
            },
            settings.tts_mms_model,
            default_zh_voice=settings.tts_default_zh_voice,
            piper_data_dir=settings.tts_piper_data_dir,
            piper_tokenizer_dir=settings.tts_piper_tokenizer_dir,
            device=settings.device,
            offline=settings.offline,
        )
    else:
        raise ModelUnavailableError(f"Unsupported TTS backend '{settings.tts_backend}'.")
    return SpeechTranslationPipeline(settings, asr, translator, tts)
