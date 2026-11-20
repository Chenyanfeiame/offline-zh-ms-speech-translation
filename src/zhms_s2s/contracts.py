from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class Timings(BaseModel):
    audio_decode_ms: float = 0.0
    asr_ms: float = 0.0
    enhancement_ms: float = 0.0
    mt_ms: float = 0.0
    tts_ms: float = 0.0
    total_ms: float = 0.0
    audio_duration_s: float = 0.0
    rtf: float | None = None


class ASRResult(BaseModel):
    source_language: str
    language_confidence: float | None = None
    language_mode: str
    raw_text: str
    text: str
    duration_s: float
    elapsed_ms: float
    # Internal hand-off from ASR to MT. It is deliberately excluded from the
    # standalone ASR API; raw_text always preserves the first-pass transcript.
    translation_text: str | None = Field(default=None, exclude=True)
    context_mode: str | None = None
    context_term_count: int = 0
    user_context_terms: list[str] = Field(default_factory=list)
    asr_passes: int = 1
    enhancement_ms: float = 0.0
    detected_entities: list[dict] = Field(default_factory=list)
    number_normalizations: list[dict] = Field(default_factory=list)
    correction_log: list[dict] = Field(default_factory=list)
    entity_rescue_log: dict = Field(default_factory=dict)


class TranslationRequest(BaseModel):
    text: str = Field(min_length=1, max_length=5000)
    source_language: str
    target_language: str | None = None

    @model_validator(mode="after")
    def different_languages(self) -> "TranslationRequest":
        if self.target_language and self.target_language == self.source_language:
            raise ValueError("source_language and target_language must differ")
        return self


class TranslationResult(BaseModel):
    source_language: str
    target_language: str
    source_text: str
    translated_text: str
    pivot_text: str | None = None
    elapsed_ms: float
    actual_mt_input: str
    actual_mt_request_messages: list[dict] = Field(default_factory=list)
    mt_raw_output: str
    model_id: str
    session_id: str | None = None
    mt_request_id: str | None = None
    mt_cache_prompt: bool | None = None


class TTSRequest(BaseModel):
    text: str = Field(min_length=1, max_length=5000)
    language: str
    voice_id: str | None = None


class TTSResult(BaseModel):
    language: str
    voice_id: str
    display_text: str
    spoken_text: str
    semantic_spans: list[dict] = Field(default_factory=list)
    audio_path: str
    audio_url: str
    duration_s: float
    elapsed_ms: float


class EditedSpeechRequest(BaseModel):
    request_id: str = Field(min_length=32, max_length=36)
    parent_request_id: str = Field(min_length=32, max_length=36)
    raw_asr_text: str = Field(min_length=1, max_length=5000)
    edited_asr_text: str = Field(min_length=1, max_length=5000)
    source_language: str
    target_language: str | None = None
    voice_id: str | None = None


class EditedSpeechResult(BaseModel):
    request_id: str
    parent_request_id: str
    source_language: str
    target_language: str
    voice_id: str
    raw_asr_text: str
    edited_asr_text: str
    asr_was_edited: bool
    correction_log: list[dict] = Field(default_factory=list)
    actual_mt_input: str
    actual_mt_request_messages: list[dict] = Field(default_factory=list)
    mt_raw_output: str
    model_id: str
    session_id: str | None = None
    mt_request_id: str | None = None
    mt_cache_prompt: bool | None = None
    translated_text: str
    pivot_text: str | None = None
    tts_display_text: str
    tts_spoken_text: str
    tts_semantic_spans: list[dict] = Field(default_factory=list)
    output_audio_path: str
    output_audio_url: str
    timings: Timings


class SpeechTranslationResult(BaseModel):
    request_id: str
    source_language: str
    target_language: str
    language_confidence: float | None = None
    language_mode: str
    voice_id: str
    raw_asr_text: str
    enhanced_asr_text: str
    asr_text: str
    asr_context_mode: str | None = None
    asr_context_term_count: int = 0
    user_context_terms: list[str] = Field(default_factory=list)
    asr_passes: int = 1
    detected_entities: list[dict] = Field(default_factory=list)
    number_normalizations: list[dict] = Field(default_factory=list)
    correction_log: list[dict] = Field(default_factory=list)
    entity_rescue_log: dict = Field(default_factory=dict)
    actual_mt_input: str
    actual_mt_request_messages: list[dict] = Field(default_factory=list)
    mt_raw_output: str
    model_id: str
    session_id: str | None = None
    mt_request_id: str | None = None
    mt_cache_prompt: bool | None = None
    translated_text: str
    pivot_text: str | None = None
    tts_display_text: str
    tts_spoken_text: str
    tts_semantic_spans: list[dict] = Field(default_factory=list)
    output_audio_path: str
    output_audio_url: str
    timings: Timings


class ErrorBody(BaseModel):
    code: str
    message: str
    detail: dict = Field(default_factory=dict)
