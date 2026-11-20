from pathlib import Path

import numpy as np

from zhms_s2s.audio import AudioData
from zhms_s2s.backends.asr import (
    PauseSegmentedASR,
    TransformersWhisperASR,
    _find_pause_boundaries,
    _mlx_translation_hint,
    normalize_asr_text,
)
from zhms_s2s.backends.interfaces import ASRTranscript


def test_chinese_asr_text_is_normalized_to_simplified():
    assert normalize_asr_text("為什麼這麼慢？聲音留在電腦裡。", "zh") == (
        "为什么这么慢？声音留在电脑里。"
    )


def test_malay_asr_text_is_unchanged():
    text = "Kenapa begitu lambat?"
    assert normalize_asr_text(text, "ms") == text


def _two_sentence_audio(silence_s: float = 0.5) -> AudioData:
    sample_rate = 16_000
    first = np.full(int(1.5 * sample_rate), 0.12, dtype=np.float32)
    silence = np.zeros(int(silence_s * sample_rate), dtype=np.float32)
    second = np.full(int(1.5 * sample_rate), -0.12, dtype=np.float32)
    samples = np.concatenate((first, silence, second))
    return AudioData(samples, sample_rate, len(samples) / sample_rate)


def test_pause_detector_only_uses_clear_sentence_gaps():
    assert len(_find_pause_boundaries(_two_sentence_audio(0.5))) == 1
    assert _find_pause_boundaries(_two_sentence_audio(0.15)) == []


class _SequentialASR:
    def __init__(self) -> None:
        self.outputs = iter(
            (
                "Nombor pesanan saya ialah A7-392",
                "Jangan buat bayaran dua kali",
            )
        )

    def transcribe(self, _audio, language):
        return ASRTranscript(next(self.outputs), language, None, "manual")

    def status(self):
        return {"backend": "fake", "model_present": True}


class _BatchASR:
    def __init__(self) -> None:
        self.requested_num_beams = None

    def transcribe_batch(self, _audios, language, *, num_beams=None):
        self.requested_num_beams = num_beams
        return [
            ASRTranscript("Ayat pertama", language, None, "manual"),
            ASRTranscript("Ayat kedua", language, None, "manual"),
        ]

    def status(self):
        return {"backend": "fake-batch", "model_present": True}


def test_malay_pause_segmentation_restores_sentence_boundaries():
    backend = PauseSegmentedASR(_SequentialASR(), language="ms")

    result = backend.transcribe(_two_sentence_audio(), "ms")

    assert result.text == (
        "Nombor pesanan saya ialah A7-392. Jangan buat bayaran dua kali."
    )
    assert result.translation_text == result.text


def test_pause_segmented_batch_uses_fast_segment_beam():
    core = _BatchASR()
    backend = PauseSegmentedASR(
        core,
        language="ms",
        segmented_num_beams=1,
    )

    result = backend.transcribe(_two_sentence_audio(), "ms")

    assert core.requested_num_beams == 1
    assert result.text == "Ayat pertama. Ayat kedua."
    assert backend.status()["segmented_num_beams"] == 1


def test_transformers_beam_override_keeps_unsplit_quality_default():
    backend = TransformersWhisperASR(Path("unused"), num_beams=2)

    assert backend._generation_options("ms")["num_beams"] == 2
    assert backend._generation_options("ms", num_beams=1)["num_beams"] == 1


def test_mlx_translation_hint_keeps_visible_text_but_marks_a_tail_clause():
    text = "我不知道这个药会不会坏掉他说没事"
    hint = _mlx_translation_hint(
        text,
        "zh",
        [
            {"start": 0.0, "text": "我不知道这个药会不会坏掉"},
            {"start": 3.1, "text": "他说没事"},
        ],
        [],
    )

    assert hint == "我不知道这个药会不会坏掉；他说没事"
    assert text == "我不知道这个药会不会坏掉他说没事"
