from __future__ import annotations


class DemoError(Exception):
    """Base error with a stable API error code and HTTP status."""

    code = "DEMO_ERROR"
    status_code = 500

    def __init__(self, message: str, *, detail: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail or {}


class InvalidLanguageError(DemoError):
    code = "INVALID_LANGUAGE"
    status_code = 422


class InvalidAudioError(DemoError):
    code = "INVALID_AUDIO"
    status_code = 422


class InvalidVoiceError(DemoError):
    code = "INVALID_VOICE"
    status_code = 422


class InvalidContextError(DemoError):
    code = "INVALID_CONTEXT"
    status_code = 422


class InvalidRequestIdError(DemoError):
    code = "INVALID_REQUEST_ID"
    status_code = 422


class ModelUnavailableError(DemoError):
    code = "MODEL_UNAVAILABLE"
    status_code = 503


class InferenceError(DemoError):
    code = "INFERENCE_FAILED"
    status_code = 500
