from __future__ import annotations

from pathlib import Path

from zhms_s2s.config import Settings
from zhms_s2s.backends.interfaces import ASRTranscript
from zhms_s2s.pipeline import SpeechTranslationPipeline

from .fakes import FakeASR, FakeTTS, FakeTranslator
from .test_audio import wav_bytes


def make_pipeline(tmp_path):
    settings = Settings(project_root=tmp_path, outputs_dir=tmp_path / "outputs")
    settings.outputs_dir.mkdir()
    return SpeechTranslationPipeline(settings, FakeASR(), FakeTranslator(), FakeTTS())


def test_both_directions(tmp_path):
    pipeline = make_pipeline(tmp_path)
    zh_ms = pipeline.run(wav_bytes(), filename="zh.wav", source_language="zh")
    ms_zh = pipeline.run(wav_bytes(), filename="ms.wav", source_language="ms")
    assert (zh_ms.source_language, zh_ms.target_language) == ("zh", "ms")
    assert zh_ms.translated_text == "selamat datang"
    assert (ms_zh.source_language, ms_zh.target_language) == ("ms", "zh")
    assert ms_zh.pivot_text == "welcome"
    assert (tmp_path / "outputs" / zh_ms.output_audio_path.split("/")[-1]).is_file()
    assert zh_ms.timings.rtf is not None


def test_direct_baseline_mt_uses_raw_asr_and_ignores_hidden_hint(tmp_path):
    class HintASR:
        def transcribe(self, _audio, _source_language):
            return ASRTranscript(
                text="我不知道药会不会坏掉他说没事",
                language="zh",
                language_confidence=None,
                language_mode="manual",
                translation_text="我不知道药会不会坏掉；他说没事",
            )

        def status(self):
            return {"backend": "fake", "model_present": True}

    class CapturingTranslator(FakeTranslator):
        seen_text = None

        def translate(self, text, source_language, target_language):
            self.seen_text = text
            return super().translate(text, source_language, target_language)

    settings = Settings(project_root=tmp_path, outputs_dir=tmp_path / "outputs")
    settings.outputs_dir.mkdir()
    translator = CapturingTranslator()
    pipeline = SpeechTranslationPipeline(settings, HintASR(), translator, FakeTTS())

    result = pipeline.run(wav_bytes(), filename="zh.wav", source_language="zh")

    assert result.asr_text == "我不知道药会不会坏掉他说没事"
    assert translator.seen_text == "我不知道药会不会坏掉他说没事"
    assert result.actual_mt_input == result.raw_asr_text
    assert result.enhanced_asr_text == result.raw_asr_text
    assert result.correction_log == []


def test_pipeline_applies_one_general_context_pass_and_enhances_output(tmp_path):
    class ContextASR(FakeASR):
        seen_context = None
        calls = 0

        def transcribe_with_context(self, _audio, _source_language, *, context):
            self.calls += 1
            self.seen_context = context
            return ASRTranscript(
                text=(
                    "Sila tiba di KL Central pada pukul tiga petang, "
                    "dua puluh lima November."
                ),
                language="ms",
                language_confidence=None,
                language_mode="manual",
            )

    class CapturingTranslator(FakeTranslator):
        seen_text = None

        def translate(self, text, source_language, target_language):
            self.seen_text = text
            return "欢迎", "welcome"

    settings = Settings(project_root=tmp_path, outputs_dir=tmp_path / "outputs")
    settings.outputs_dir.mkdir()
    settings.baseline_direct_pipeline = False
    settings.entity_enhancement_enabled = True
    settings.entity_prompt_enabled = True
    asr = ContextASR()
    translator = CapturingTranslator()
    pipeline = SpeechTranslationPipeline(settings, asr, translator, FakeTTS())

    result = pipeline.run(
        wav_bytes(),
        filename="ms.wav",
        source_language="ms",
        user_context="Restoran Rebung, MH123",
    )

    assert "Restoran Rebung" in asr.seen_context
    assert "KL Sentral" in asr.seen_context
    assert asr.calls == 1
    assert result.asr_passes == 1
    assert result.raw_asr_text.startswith("Sila tiba di KL Central")
    assert result.enhanced_asr_text == (
        "Sila tiba di KL Sentral pada pukul 3 petang, 25 November."
    )
    assert translator.seen_text == result.enhanced_asr_text
    assert result.asr_context_mode == "general-entity-prompt-v1"
    assert result.user_context_terms == ["Restoran Rebung", "MH123"]


def test_chinese_asr_does_not_receive_malay_entity_prompt(tmp_path):
    class ChineseASR(FakeASR):
        calls = 0

        def transcribe(self, _audio, _source_language):
            self.calls += 1
            return ASRTranscript(
                text="我的航班从北京飞往槟城。",
                language="zh",
                language_confidence=None,
                language_mode="manual",
            )

        def transcribe_with_context(self, *_args, **_kwargs):
            raise AssertionError("Chinese ASR must never receive the Malay prompt")

    settings = Settings(project_root=tmp_path, outputs_dir=tmp_path / "outputs")
    settings.outputs_dir.mkdir()
    asr = ChineseASR()
    pipeline = SpeechTranslationPipeline(
        settings, asr, FakeTranslator(), FakeTTS()
    )

    result = pipeline.run(
        wav_bytes(), filename="zh.wav", source_language="zh"
    )

    assert asr.calls == 1
    assert result.asr_text == "我的航班从北京飞往槟城。"
    assert result.asr_passes == 1
    assert result.asr_context_mode is None


def test_pipeline_writes_raw_enhanced_correction_and_translation_audit(tmp_path):
    pipeline = make_pipeline(tmp_path)

    result = pipeline.run(wav_bytes(), filename="ms.wav", source_language="ms")

    audit = (tmp_path / "outputs" / "pipeline_audit.jsonl").read_text(
        encoding="utf-8"
    )
    assert result.request_id in audit
    assert '"raw_asr_output"' in audit
    assert '"enhanced_asr_output"' in audit
    assert '"correction_log"' in audit
    assert '"actual_mt_input"' in audit
    assert '"actual_mt_request_messages"' in audit
    assert '"mt_raw_output"' in audit
    assert '"model_id"' in audit
    assert '"session_id"' in audit
    assert '"final_translation"' in audit
    assert '"tts_display_text"' in audit
    assert '"tts_spoken_text"' in audit
    assert '"tts_semantic_spans"' in audit


def test_tts_pipeline_keeps_display_text_and_sends_semantic_spoken_text(tmp_path):
    class CapturingTTS(FakeTTS):
        seen_text = None

        def synthesize(self, text, language, output_path, voice_id=None):
            self.seen_text = text
            return super().synthesize(text, language, output_path, voice_id)

    settings = Settings(project_root=tmp_path, outputs_dir=tmp_path / "outputs")
    settings.outputs_dir.mkdir()
    tts = CapturingTTS()
    pipeline = SpeechTranslationPipeline(
        settings, FakeASR(), FakeTranslator(), tts
    )
    display = (
        "我的电话号码是013-4567890，订单号是1234，机票价格是1234令吉，"
        "航班MH1234预计延误45分钟。"
    )

    result = pipeline.synthesize_text(display, "zh")

    assert result.display_text == display
    assert result.spoken_text == tts.seen_text
    assert "订单号是一二三四" in result.spoken_text
    assert "机票价格是一千二百三十四令吉" in result.spoken_text
    assert [span["semantic_type"] for span in result.semantic_spans] == [
        "telephone",
        "identifier",
        "money",
        "flight_identifier",
        "duration",
    ]


def test_direct_baseline_bypasses_context_and_entity_enhancement(tmp_path):
    class ContextMustNotRunASR(FakeASR):
        def transcribe(self, _audio, _source_language):
            return ASRTranscript(
                text="Sila tiba di KL Central pada dua puluh lima November.",
                language="ms",
                language_confidence=None,
                language_mode="manual",
            )

        def transcribe_with_context(self, *_args, **_kwargs):
            raise AssertionError("direct baseline must not send an ASR prompt")

    class CapturingTranslator(FakeTranslator):
        seen_text = None

        def translate(self, text, source_language, target_language):
            self.seen_text = text
            return super().translate(text, source_language, target_language)

    settings = Settings(project_root=tmp_path, outputs_dir=tmp_path / "outputs")
    settings.outputs_dir.mkdir()
    settings.entity_prompt_enabled = True
    settings.entity_enhancement_enabled = True
    translator = CapturingTranslator()
    pipeline = SpeechTranslationPipeline(
        settings, ContextMustNotRunASR(), translator, FakeTTS()
    )

    result = pipeline.run(
        wav_bytes(),
        filename="ms.wav",
        source_language="ms",
        user_context="KL Sentral, A7392",
    )

    assert result.raw_asr_text == (
        "Sila tiba di KL Central pada dua puluh lima November."
    )
    assert result.enhanced_asr_text == result.raw_asr_text
    assert result.detected_entities == []
    assert result.number_normalizations == []
    assert result.correction_log == []
    assert result.asr_context_term_count == 0
    assert translator.seen_text == result.raw_asr_text


def test_user_edited_asr_is_the_exact_mt_input_and_generates_new_tts(tmp_path):
    class CapturingTranslator(FakeTranslator):
        seen_text = None

        def translate(self, text, source_language, target_language):
            self.seen_text = text
            return super().translate(text, source_language, target_language)

    settings = Settings(project_root=tmp_path, outputs_dir=tmp_path / "outputs")
    settings.outputs_dir.mkdir()
    translator = CapturingTranslator()
    pipeline = SpeechTranslationPipeline(
        settings, FakeASR(), translator, FakeTTS()
    )

    result = pipeline.translate_edited_asr(
        request_id="aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
        parent_request_id="bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb",
        raw_asr_text="Hi, Salamat datang.",
        edited_asr_text="Hai, selamat datang.",
        source_language="ms",
        target_language="zh",
    )

    assert result.asr_was_edited is True
    assert result.actual_mt_input == "Hai, selamat datang."
    assert translator.seen_text == "Hai, selamat datang."
    assert result.parent_request_id == "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"
    assert Path(result.output_audio_path).is_file()
    audit = (tmp_path / "outputs" / "pipeline_audit.jsonl").read_text(
        encoding="utf-8"
    )
    assert '"workflow_stage": "user-edited-asr-to-mt-tts"' in audit
    assert '"edited_asr_output": "Hai, selamat datang."' in audit


def test_selective_entity_rescue_preserves_raw_and_only_sends_gated_text_to_mt(tmp_path):
    class ContextASR(FakeASR):
        def transcribe(self, _audio, _source_language):
            return ASRTranscript(
                text="Number pasangan saya, jangan buat bayaran dua kali.",
                language="ms",
                language_confidence=None,
                language_mode="manual",
            )

        def transcribe_with_context(self, _audio, _source_language, *, context):
            assert context == "nombor pesanan"
            return ASRTranscript(
                text="Nombor pesanan saya, jangan buat bayaran dua kali.",
                language="ms",
                language_confidence=None,
                language_mode="manual",
            )

    class CapturingTranslator(FakeTranslator):
        seen_text = None

        def translate(self, text, source_language, target_language):
            self.seen_text = text
            return super().translate(text, source_language, target_language)

    settings = Settings(project_root=tmp_path, outputs_dir=tmp_path / "outputs")
    settings.outputs_dir.mkdir()
    settings.selective_entity_rescue_enabled = True
    translator = CapturingTranslator()
    pipeline = SpeechTranslationPipeline(
        settings, ContextASR(), translator, FakeTTS()
    )

    result = pipeline.run(wav_bytes(), filename="ms.wav", source_language="ms")

    assert result.raw_asr_text.startswith("Number pasangan")
    assert result.enhanced_asr_text.startswith("Nombor pesanan")
    assert result.actual_mt_input == result.enhanced_asr_text
    assert translator.seen_text == result.enhanced_asr_text
    assert result.asr_passes == 2
    assert result.entity_rescue_log["accepted_contextual_pass"] is True
    assert "jangan" in result.enhanced_asr_text.casefold()


def test_chinese_selective_entity_rescue_reaches_mt_without_losing_negation(tmp_path):
    class ContextASR(FakeASR):
        def transcribe(self, _audio, _source_language):
            return ASRTranscript(
                text="请不要抵达吉隆坡中央车占。",
                language="zh",
                language_confidence=None,
                language_mode="manual",
            )

        def transcribe_with_context(self, _audio, _source_language, *, context):
            assert context == "吉隆坡中央车站"
            return ASRTranscript(
                text="请不要抵达吉隆坡中央车站。",
                language="zh",
                language_confidence=None,
                language_mode="manual",
            )

    class CapturingTranslator(FakeTranslator):
        seen_text = None

        def translate(self, text, source_language, target_language):
            self.seen_text = text
            return super().translate(text, source_language, target_language)

    settings = Settings(project_root=tmp_path, outputs_dir=tmp_path / "outputs")
    settings.outputs_dir.mkdir()
    settings.selective_entity_rescue_enabled = True
    translator = CapturingTranslator()
    pipeline = SpeechTranslationPipeline(
        settings, ContextASR(), translator, FakeTTS()
    )

    result = pipeline.run(wav_bytes(), filename="zh.wav", source_language="zh")

    assert result.raw_asr_text.endswith("中央车占。")
    assert result.enhanced_asr_text.endswith("中央车站。")
    assert result.actual_mt_input == result.enhanced_asr_text
    assert translator.seen_text == result.enhanced_asr_text
    assert result.asr_passes == 2
    assert result.entity_rescue_log["accepted_contextual_pass"] is True
    assert "不要" in result.enhanced_asr_text
