from __future__ import annotations

from fastapi.testclient import TestClient

from zhms_s2s.api import create_app
from zhms_s2s.config import Settings
from zhms_s2s.pipeline import SpeechTranslationPipeline

from .fakes import FakeASR, FakeTTS, FakeTranslator
from .test_audio import wav_bytes


def client(tmp_path):
    settings = Settings(project_root=tmp_path, outputs_dir=tmp_path / "outputs")
    settings.outputs_dir.mkdir()
    pipeline = SpeechTranslationPipeline(settings, FakeASR(), FakeTranslator(), FakeTTS())
    return TestClient(create_app(settings, pipeline))


def test_health_and_normal_requests(tmp_path):
    with client(tmp_path) as api:
        home = api.get("/")
        assert home.status_code == 200
        assert "开始离线翻译" in home.text
        assert "ASR 识别结果（可修改）" in home.text
        assert "稳定直连模式与音色" not in home.text
        assert "Raw ASR → MT → TTS" not in home.text
        assert "实体、数字、别名修正与所有 ASR/MT 提示词已默认旁路" not in home.text
        assert api.get("/health").json()["ready"] is True
        warmed = api.post("/warmup")
        assert warmed.status_code == 200
        assert warmed.json()["status"] == "ready"
        translated = api.post(
            "/translate",
            json={"text": "你好", "source_language": "zh", "target_language": "ms"},
        )
        assert translated.status_code == 200
        assert translated.json()["translated_text"] == "selamat datang"
        speech = api.post(
            "/speech-translate",
            files={"file": ("sample.wav", wav_bytes(), "audio/wav")},
            data={
                "source_language": "ms",
                "user_context": "Restoran Rebung, nasi kerabu",
                "client_request_id": "de305d54-75b4-431b-adb2-eb6b9e546014",
            },
        )
        assert speech.status_code == 200
        assert speech.json()["target_language"] == "zh"
        assert speech.json()["raw_asr_text"] == "selamat datang"
        assert speech.json()["enhanced_asr_text"] == "selamat datang"
        assert speech.json()["request_id"] == "de305d54-75b4-431b-adb2-eb6b9e546014"
        assert speech.json()["actual_mt_input"] == "selamat datang"
        assert speech.json()["session_id"] is None
        edited = api.post(
            "/edited-speech-translate",
            json={
                "request_id": "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
                "parent_request_id": speech.json()["request_id"],
                "raw_asr_text": speech.json()["raw_asr_text"],
                "edited_asr_text": "selamat datang ke demo",
                "source_language": "ms",
                "target_language": "zh",
            },
        )
        assert edited.status_code == 200
        assert edited.json()["request_id"] == "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
        assert edited.json()["parent_request_id"] == speech.json()["request_id"]
        assert edited.json()["actual_mt_input"] == "selamat datang ke demo"
        assert edited.json()["asr_was_edited"] is True
        assert edited.json()["output_audio_url"].endswith(
            "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa_zh.wav"
        )
        asr = api.post(
            "/asr",
            files={"file": ("sample.wav", wav_bytes(), "audio/wav")},
            data={"source_language": "ms"},
        )
        assert asr.json()["raw_asr_text"] == "selamat datang"
        assert asr.json()["enhanced_asr_text"] == "selamat datang"


def test_request_errors(tmp_path):
    with client(tmp_path) as api:
        missing = api.post("/speech-translate")
        assert missing.status_code == 422
        wrong_language = api.post(
            "/translate", json={"text": "x", "source_language": "en"}
        )
        assert wrong_language.status_code == 422
        assert wrong_language.json()["error"]["code"] == "INVALID_LANGUAGE"
        same_language = api.post(
            "/translate",
            json={"text": "x", "source_language": "zh", "target_language": "zh"},
        )
        assert same_language.status_code == 422
        assert same_language.json()["error"]["code"] == "VALIDATION_ERROR"
        bad_audio = api.post(
            "/asr", files={"file": ("bad.mp3", b"not audio", "audio/mpeg")}
        )
        assert bad_audio.status_code == 422
        assert bad_audio.json()["error"]["code"] == "INVALID_AUDIO"
        ignored_context = api.post(
            "/speech-translate",
            files={"file": ("sample.wav", wav_bytes(), "audio/wav")},
            data={
                "source_language": "ms",
                "user_context": ",".join(f"term-{index}" for index in range(17)),
            },
        )
        assert ignored_context.status_code == 200
        assert ignored_context.json()["asr_context_term_count"] == 0
        invalid_request_id = api.post(
            "/speech-translate",
            files={"file": ("sample.wav", wav_bytes(), "audio/wav")},
            data={"source_language": "ms", "client_request_id": "not-a-uuid"},
        )
        assert invalid_request_id.status_code == 422
        assert invalid_request_id.json()["error"]["code"] == "INVALID_REQUEST_ID"
