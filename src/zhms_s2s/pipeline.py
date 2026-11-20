from __future__ import annotations

import json
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .audio import AudioData, decode_audio_bytes, preprocess_audio
from .backends.interfaces import ASRBackend, TTSBackend, TranslationBackend
from .backends.tts import build_tts_plan
from .chinese_phonetics import ChinesePhoneticEncoder
from .config import Settings, target_for, validate_language
from .contracts import (
    ASRResult,
    EditedSpeechResult,
    SpeechTranslationResult,
    TTSResult,
    Timings,
    TranslationResult,
)
from .errors import InvalidContextError, InvalidRequestIdError
from .enhancement import EntityNumberEnhancer, parse_user_terms
from .entity_rescue import SelectiveEntityRescuer


class SpeechTranslationPipeline:
    def __init__(
        self,
        settings: Settings,
        asr: ASRBackend,
        translator: TranslationBackend,
        tts: TTSBackend,
    ) -> None:
        self.settings = settings
        self.asr_backend = asr
        self.translation_backend = translator
        self.tts_backend = tts
        lexicon_path = Path(settings.entity_lexicon)
        if not lexicon_path.is_absolute():
            lexicon_path = settings.project_root / lexicon_path
        if not lexicon_path.is_file():
            lexicon_path = Path(__file__).resolve().parents[2] / "data/entity_lexicon.json"
        self._lexicon_path = lexicon_path
        entity_matcher = (
            EntityNumberEnhancer(lexicon_path)
            if (
                settings.selective_entity_rescue_enabled
                or not settings.baseline_direct_pipeline
            )
            else None
        )
        self.enhancer = None if settings.baseline_direct_pipeline else entity_matcher
        phonetic_encoder = ChinesePhoneticEncoder(
            settings.tts_piper_data_dir / "g2pW",
            settings.tts_piper_tokenizer_dir,
        )
        self.entity_rescuer = (
            SelectiveEntityRescuer(
                entity_matcher,
                max_candidates=settings.selective_entity_rescue_max_candidates,
                max_prompt_characters=(
                    settings.selective_entity_rescue_max_prompt_characters
                ),
                phonetic_encoder=phonetic_encoder,
            )
            if settings.selective_entity_rescue_enabled
            and entity_matcher is not None
            else None
        )
        self._audit_lock = threading.Lock()

    def decode(self, data: bytes, filename: str | None) -> tuple[AudioData, float]:
        started = time.perf_counter()
        audio = decode_audio_bytes(
            data,
            filename=filename,
            max_seconds=self.settings.max_audio_seconds,
        )
        if self.settings.asr_preprocess_audio:
            audio = preprocess_audio(audio)
        return audio, (time.perf_counter() - started) * 1000

    def transcribe_audio(
        self,
        audio: AudioData,
        source_language: str | None,
        *,
        user_context: str | None = None,
    ) -> ASRResult:
        if self.settings.baseline_direct_pipeline:
            user_terms: tuple[str, ...] = ()
            if self.entity_rescuer is not None:
                try:
                    user_terms = parse_user_terms(user_context)
                except ValueError as exc:
                    raise InvalidContextError(str(exc)) from exc
            started = time.perf_counter()
            transcript = self.asr_backend.transcribe(audio, source_language)
            first_pass_ms = (time.perf_counter() - started) * 1000
            selected_text = transcript.text
            entities: list[dict] = []
            corrections: list[dict] = []
            candidates = ()
            contextual_text = None
            contextual_accepted = False
            decision_reason = "selective-entity-rescue-disabled"
            safety_checks: dict = {}
            prompt = ""
            asr_passes = 1
            enhancement_started = time.perf_counter()
            if self.entity_rescuer is not None:
                candidates = self.entity_rescuer.retrieve(
                    transcript.text,
                    transcript.language,
                    user_terms=user_terms,
                )
                prompt = self.entity_rescuer.build_prompt(candidates)
                contextual = getattr(
                    self.asr_backend, "transcribe_with_context", None
                )
                if (
                    prompt
                    and callable(contextual)
                    and self.entity_rescuer.needs_contextual_pass(
                        transcript.text, transcript.language, candidates
                    )
                ):
                    contextual_text = contextual(
                        audio,
                        transcript.language,
                        context=prompt,
                    ).text
                    asr_passes = 2
                rescue = self.entity_rescuer.select(
                    transcript.text,
                    transcript.language,
                    candidates=candidates,
                    contextual_text=contextual_text,
                )
                selected_text = rescue.selected_text
                entities = list(rescue.entities)
                corrections = list(rescue.corrections)
                contextual_accepted = rescue.accepted_contextual_pass
                decision_reason = rescue.decision_reason
                safety_checks = rescue.safety_checks
            enhancement_ms = (time.perf_counter() - enhancement_started) * 1000
            asr_elapsed = first_pass_ms + (
                enhancement_ms if asr_passes == 2 else 0.0
            )
            return ASRResult(
                source_language=transcript.language,
                language_confidence=transcript.language_confidence,
                language_mode=transcript.language_mode,
                raw_text=transcript.text,
                text=selected_text,
                duration_s=audio.duration_s,
                elapsed_ms=asr_elapsed,
                # Only the separately gated entity-rescue result may replace
                # the first-pass transcript in otherwise-direct mode.
                translation_text=(
                    selected_text if self.entity_rescuer is not None else None
                ),
                context_mode=(
                    "selective-entity-rescue-v1" if candidates else None
                ),
                context_term_count=len(candidates),
                user_context_terms=list(user_terms),
                asr_passes=asr_passes,
                enhancement_ms=enhancement_ms,
                detected_entities=entities,
                number_normalizations=[],
                correction_log=corrections,
                entity_rescue_log={
                    "candidates": [item.as_dict() for item in candidates],
                    "prompt": prompt,
                    "contextual_text": contextual_text,
                    "accepted_contextual_pass": contextual_accepted,
                    "decision_reason": decision_reason,
                    "safety_checks": safety_checks,
                },
            )

        try:
            user_terms = parse_user_terms(user_context)
        except ValueError as exc:
            raise InvalidContextError(str(exc)) from exc
        prompt = ""
        prompt_terms: tuple[str, ...] = ()
        if self.settings.entity_prompt_enabled and self.enhancer is not None:
            prompt, prompt_terms = self.enhancer.build_prompt(
                "ms",
                user_terms=user_terms,
                max_terms=self.settings.entity_prompt_max_terms,
                max_characters=self.settings.entity_prompt_max_characters,
            )
        elif user_terms:
            # Explicit request-specific terms are much narrower than the
            # global lexicon and remain useful for known booking codes/names.
            prompt_terms = user_terms[: self.settings.entity_prompt_max_terms]
            prompt = " ".join(prompt_terms)
            prompt = prompt[: self.settings.entity_prompt_max_characters].strip()
        started = time.perf_counter()
        contextual = getattr(self.asr_backend, "transcribe_with_context", None)
        transcript = (
            contextual(audio, source_language, context=prompt)
            if prompt and callable(contextual)
            else self.asr_backend.transcribe(audio, source_language)
        )
        asr_elapsed = (time.perf_counter() - started) * 1000
        context_applied = bool(
            prompt and callable(contextual)
        )
        enhancement_started = time.perf_counter()
        if self.enhancer is None:
            raise RuntimeError("Enhancer is unavailable outside direct baseline mode.")
        enhancement = self.enhancer.enhance(
            transcript.text,
            transcript.language,
            enable_entities=self.settings.entity_enhancement_enabled,
            enable_numbers=self.settings.entity_enhancement_enabled,
        )
        enhanced_translation_text = None
        if transcript.translation_text:
            enhanced_translation_text = self.enhancer.enhance(
                transcript.translation_text,
                transcript.language,
                enable_entities=self.settings.entity_enhancement_enabled,
                enable_numbers=self.settings.entity_enhancement_enabled,
            ).enhanced_text
        enhancement_elapsed = (time.perf_counter() - enhancement_started) * 1000
        return ASRResult(
            source_language=transcript.language,
            language_confidence=transcript.language_confidence,
            language_mode=transcript.language_mode,
            raw_text=transcript.text,
            text=enhancement.enhanced_text,
            duration_s=audio.duration_s,
            elapsed_ms=asr_elapsed,
            translation_text=enhanced_translation_text,
            context_mode="general-entity-prompt-v1" if context_applied else None,
            context_term_count=len(prompt_terms) if context_applied else 0,
            user_context_terms=list(user_terms) if context_applied else [],
            asr_passes=1,
            enhancement_ms=enhancement_elapsed,
            detected_entities=list(enhancement.entities),
            number_normalizations=list(enhancement.number_normalizations),
            correction_log=list(enhancement.corrections),
        )

    def translate_text(
        self,
        text: str,
        source_language: str,
        target_language: str | None = None,
        *,
        request_id: str | None = None,
    ) -> TranslationResult:
        source = validate_language(source_language)
        target = validate_language(target_language) if target_language else target_for(source)
        started = time.perf_counter()
        traced_translate = getattr(
            self.translation_backend, "translate_with_trace", None
        )
        if callable(traced_translate):
            translated, pivot, trace = traced_translate(
                text,
                source,
                target,
                request_id=request_id,
            )
        else:
            translated, pivot = self.translation_backend.translate(text, source, target)
            status = self.translation_backend.status()
            trace = {
                "actual_mt_request_messages": [],
                "mt_raw_output": translated,
                "model_id": str(status.get("backend", "unknown")),
                "session_id": None,
                "mt_request_id": request_id,
                "cache_prompt": None,
            }
        elapsed = (time.perf_counter() - started) * 1000
        return TranslationResult(
            source_language=source,
            target_language=target,
            source_text=text,
            translated_text=translated,
            pivot_text=pivot,
            elapsed_ms=elapsed,
            actual_mt_input=text,
            actual_mt_request_messages=list(
                trace.get("actual_mt_request_messages", [])
            ),
            mt_raw_output=str(trace.get("mt_raw_output", translated)),
            model_id=str(trace.get("model_id", "unknown")),
            session_id=trace.get("session_id"),
            mt_request_id=trace.get("mt_request_id"),
            mt_cache_prompt=trace.get("cache_prompt"),
        )

    def synthesize_text(
        self,
        text: str,
        language: str,
        *,
        request_id: str | None = None,
        voice_id: str | None = None,
    ) -> TTSResult:
        language = validate_language(language)
        request_id = request_id or uuid.uuid4().hex
        filename = f"{request_id}_{language}.wav"
        output_path = self.settings.outputs_dir / filename
        resolver = getattr(self.tts_backend, "resolve_voice", None)
        resolved_voice = (
            resolver(language, voice_id) if callable(resolver) else voice_id or "default"
        )
        tts_plan = build_tts_plan(text, language)
        started = time.perf_counter()
        duration = self.tts_backend.synthesize(
            tts_plan.spoken_text, language, output_path, resolved_voice
        )
        elapsed = (time.perf_counter() - started) * 1000
        return TTSResult(
            language=language,
            voice_id=resolved_voice,
            display_text=tts_plan.display_text,
            spoken_text=tts_plan.spoken_text,
            semantic_spans=[span.as_dict() for span in tts_plan.semantic_spans],
            audio_path=str(output_path),
            audio_url=f"/outputs/{filename}",
            duration_s=duration,
            elapsed_ms=elapsed,
        )

    @staticmethod
    def validate_request_id(request_id: str | None) -> str:
        if request_id:
            try:
                uuid.UUID(request_id)
            except (ValueError, AttributeError) as exc:
                raise InvalidRequestIdError(
                    "request_id must be a valid UUID."
                ) from exc
            return request_id
        return uuid.uuid4().hex

    def translate_edited_asr(
        self,
        *,
        request_id: str,
        parent_request_id: str,
        raw_asr_text: str,
        edited_asr_text: str,
        source_language: str,
        target_language: str | None = None,
        voice_id: str | None = None,
    ) -> EditedSpeechResult:
        """Re-run MT and TTS from an explicit user edit of the ASR text."""

        request_id = self.validate_request_id(request_id)
        parent_request_id = self.validate_request_id(parent_request_id)
        raw_text = raw_asr_text.strip()
        edited_text = edited_asr_text.strip()
        if not raw_text or not edited_text:
            raise InvalidContextError("ASR text cannot be empty.")
        source = validate_language(source_language)
        target = (
            validate_language(target_language)
            if target_language
            else target_for(source)
        )
        total_started = time.perf_counter()
        translation = self.translate_text(
            edited_text,
            source,
            target,
            request_id=request_id,
        )
        tts = self.synthesize_text(
            translation.translated_text,
            target,
            request_id=request_id,
            voice_id=voice_id,
        )
        total_ms = (time.perf_counter() - total_started) * 1000
        was_edited = edited_text != raw_text
        corrections = (
            [
                {
                    "correction_type": "user_edit",
                    "original": raw_text,
                    "replacement": edited_text,
                    "basis": "explicit-user-edit-in-demo",
                }
            ]
            if was_edited
            else []
        )
        result = EditedSpeechResult(
            request_id=request_id,
            parent_request_id=parent_request_id,
            source_language=source,
            target_language=target,
            voice_id=tts.voice_id,
            raw_asr_text=raw_text,
            edited_asr_text=edited_text,
            asr_was_edited=was_edited,
            correction_log=corrections,
            actual_mt_input=translation.actual_mt_input,
            actual_mt_request_messages=translation.actual_mt_request_messages,
            mt_raw_output=translation.mt_raw_output,
            model_id=translation.model_id,
            session_id=translation.session_id,
            mt_request_id=translation.mt_request_id,
            mt_cache_prompt=translation.mt_cache_prompt,
            translated_text=translation.translated_text,
            pivot_text=translation.pivot_text,
            tts_display_text=tts.display_text,
            tts_spoken_text=tts.spoken_text,
            tts_semantic_spans=tts.semantic_spans,
            output_audio_path=tts.audio_path,
            output_audio_url=tts.audio_url,
            timings=Timings(
                mt_ms=translation.elapsed_ms,
                tts_ms=tts.elapsed_ms,
                total_ms=total_ms,
            ),
        )
        self._write_edited_audit_log(result)
        return result

    def run(
        self,
        data: bytes,
        *,
        filename: str | None,
        source_language: str | None,
        target_language: str | None = None,
        voice_id: str | None = None,
        user_context: str | None = None,
        request_id: str | None = None,
    ) -> SpeechTranslationResult:
        request_id = self.validate_request_id(request_id)
        total_started = time.perf_counter()
        audio, decode_ms = self.decode(data, filename)
        asr = self.transcribe_audio(
            audio,
            source_language,
            user_context=user_context,
        )
        source = asr.source_language
        target = validate_language(target_language) if target_language else target_for(source)
        # Direct mode remains verbatim unless the independently evaluated,
        # explicitly enabled selective entity rescue produced a gated result.
        mt_input = (
            asr.raw_text
            if (
                self.settings.baseline_direct_pipeline
                and not self.settings.selective_entity_rescue_enabled
            )
            else (asr.translation_text or asr.text)
        )
        translation = self.translate_text(
            mt_input,
            source,
            target,
            request_id=request_id,
        )
        tts = self.synthesize_text(
            translation.translated_text,
            target,
            request_id=request_id,
            voice_id=voice_id,
        )
        total_ms = (time.perf_counter() - total_started) * 1000
        rtf = total_ms / 1000 / audio.duration_s if audio.duration_s else None
        result = SpeechTranslationResult(
            request_id=request_id,
            source_language=source,
            target_language=target,
            language_confidence=asr.language_confidence,
            language_mode=asr.language_mode,
            voice_id=tts.voice_id,
            raw_asr_text=asr.raw_text,
            enhanced_asr_text=asr.text,
            asr_text=asr.text,
            asr_context_mode=asr.context_mode,
            asr_context_term_count=asr.context_term_count,
            user_context_terms=asr.user_context_terms,
            asr_passes=asr.asr_passes,
            detected_entities=asr.detected_entities,
            number_normalizations=asr.number_normalizations,
            correction_log=asr.correction_log,
            entity_rescue_log=asr.entity_rescue_log,
            actual_mt_input=translation.actual_mt_input,
            actual_mt_request_messages=translation.actual_mt_request_messages,
            mt_raw_output=translation.mt_raw_output,
            model_id=translation.model_id,
            session_id=translation.session_id,
            mt_request_id=translation.mt_request_id,
            mt_cache_prompt=translation.mt_cache_prompt,
            translated_text=translation.translated_text,
            pivot_text=translation.pivot_text,
            tts_display_text=tts.display_text,
            tts_spoken_text=tts.spoken_text,
            tts_semantic_spans=tts.semantic_spans,
            output_audio_path=tts.audio_path,
            output_audio_url=tts.audio_url,
            timings=Timings(
                audio_decode_ms=decode_ms,
                asr_ms=asr.elapsed_ms,
                enhancement_ms=asr.enhancement_ms,
                mt_ms=translation.elapsed_ms,
                tts_ms=tts.elapsed_ms,
                total_ms=total_ms,
                audio_duration_s=audio.duration_s,
                rtf=rtf,
            ),
        )
        self._write_audit_log(result)
        return result

    def _write_audit_log(self, result: SpeechTranslationResult) -> None:
        if not self.settings.keep_intermediate:
            return
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "request_id": result.request_id,
            "source_language": result.source_language,
            "target_language": result.target_language,
            "raw_asr_output": result.raw_asr_text,
            "raw_asr_text": result.raw_asr_text,
            "enhanced_asr_output": result.enhanced_asr_text,
            "correction_log": result.correction_log,
            "entity_rescue_log": result.entity_rescue_log,
            "detected_entities": result.detected_entities,
            "number_normalizations": result.number_normalizations,
            "actual_mt_input": result.actual_mt_input,
            "actual_mt_request_messages": result.actual_mt_request_messages,
            "mt_raw_output": result.mt_raw_output,
            "final_translation": result.translated_text,
            "tts_display_text": result.tts_display_text,
            "tts_spoken_text": result.tts_spoken_text,
            "tts_semantic_spans": result.tts_semantic_spans,
            "model_id": result.model_id,
            "session_id": result.session_id,
            "mt_request_id": result.mt_request_id,
            "mt_cache_prompt": result.mt_cache_prompt,
            "timings": result.timings.model_dump(),
        }
        audit_path = self.settings.outputs_dir / "pipeline_audit.jsonl"
        with self._audit_lock, audit_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _write_edited_audit_log(self, result: EditedSpeechResult) -> None:
        if not self.settings.keep_intermediate:
            return
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "workflow_stage": "user-edited-asr-to-mt-tts",
            "request_id": result.request_id,
            "parent_request_id": result.parent_request_id,
            "source_language": result.source_language,
            "target_language": result.target_language,
            "raw_asr_output": result.raw_asr_text,
            "edited_asr_output": result.edited_asr_text,
            "asr_was_edited": result.asr_was_edited,
            "correction_log": result.correction_log,
            "actual_mt_input": result.actual_mt_input,
            "actual_mt_request_messages": result.actual_mt_request_messages,
            "mt_raw_output": result.mt_raw_output,
            "final_translation": result.translated_text,
            "tts_display_text": result.tts_display_text,
            "tts_spoken_text": result.tts_spoken_text,
            "tts_semantic_spans": result.tts_semantic_spans,
            "model_id": result.model_id,
            "session_id": result.session_id,
            "mt_request_id": result.mt_request_id,
            "mt_cache_prompt": result.mt_cache_prompt,
            "output_audio_path": result.output_audio_path,
            "timings": result.timings.model_dump(),
        }
        audit_path = self.settings.outputs_dir / "pipeline_audit.jsonl"
        with self._audit_lock, audit_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def status(self) -> dict:
        return {
            "asr": self.asr_backend.status(),
            "enhancement": {
                "backend": "offline-entity-number-enhancer",
                "enabled": (
                    not self.settings.baseline_direct_pipeline
                    and self.settings.entity_enhancement_enabled
                ),
                "bypassed": self.settings.baseline_direct_pipeline,
                "direct_pipeline": self.settings.baseline_direct_pipeline,
                "lexicon_path": str(self._lexicon_path),
                "entity_count": (
                    len(self.enhancer.entries)
                    if self.enhancer is not None
                    else (
                        len(self.entity_rescuer.enhancer.entries)
                        if self.entity_rescuer is not None
                        else 0
                    )
                ),
                "prompt_enabled": (
                    not self.settings.baseline_direct_pipeline
                    and self.settings.entity_prompt_enabled
                ),
                "prompt_limits": {
                    "terms": self.settings.entity_prompt_max_terms,
                    "characters": self.settings.entity_prompt_max_characters,
                },
                "scene_routing": False,
                "selective_entity_rescue": {
                    "enabled": self.entity_rescuer is not None,
                    "strategy": "transcript-retrieval-plus-semantic-safety-gates",
                    "max_candidates": self.settings.selective_entity_rescue_max_candidates,
                    "max_prompt_characters": self.settings.selective_entity_rescue_max_prompt_characters,
                    "chinese_phonetic_retrieval": (
                        self.entity_rescuer.phonetic_encoder.status()
                        if self.entity_rescuer is not None
                        and self.entity_rescuer.phonetic_encoder is not None
                        else {"available": False, "loaded": False}
                    ),
                },
            },
            "translation": self.translation_backend.status(),
            "tts": self.tts_backend.status(),
        }

    def close(self) -> None:
        """Release child processes owned by local inference backends."""
        for backend in (
            self.asr_backend,
            self.translation_backend,
            self.tts_backend,
        ):
            close = getattr(backend, "close", None)
            if callable(close):
                close()

    def warmup(self) -> dict:
        started = time.perf_counter()
        timings: dict[str, float] = {}
        for name, backend in (
            ("asr_load_ms", self.asr_backend),
            ("mt_load_ms", self.translation_backend),
            ("entity_retrieval_load_ms", self.entity_rescuer),
        ):
            step_started = time.perf_counter()
            warmup = getattr(backend, "warmup", None)
            if callable(warmup):
                warmup()
            timings[name] = (time.perf_counter() - step_started) * 1000
        step_started = time.perf_counter()
        zh_tts = self.synthesize_text(
            "本地模型预热完成。",
            "zh",
            request_id="warmup_zh",
            voice_id="xiao_ya",
        )
        ms_tts = self.synthesize_text(
            "Model tempatan sudah sedia.",
            "ms",
            request_id="warmup_ms",
            voice_id="mms-zlm",
        )
        timings["tts_load_ms"] = (time.perf_counter() - step_started) * 1000
        timings["total_ms"] = (time.perf_counter() - started) * 1000
        return {
            "status": "ready",
            "timings": timings,
            "warmup_audio_url": zh_tts.audio_url,
            "warmup_audio_urls": {
                "zh": zh_tts.audio_url,
                "ms": ms_tts.audio_url,
            },
            "components": self.status(),
        }
