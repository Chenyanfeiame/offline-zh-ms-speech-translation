from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from .config import Settings
from .contracts import EditedSpeechRequest, TTSRequest, TranslationRequest
from .errors import DemoError, InvalidAudioError
from .pipeline import SpeechTranslationPipeline
from .runtime import build_pipeline


def create_app(
    settings: Settings | None = None,
    pipeline: SpeechTranslationPipeline | None = None,
) -> FastAPI:
    settings = settings or Settings.load()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.pipeline = pipeline or build_pipeline(settings)
        try:
            yield
        finally:
            app.state.pipeline.close()

    app = FastAPI(
        title="Chinese/Malay Offline Speech Translation",
        version="0.1.0",
        description="Local ASR -> MT -> TTS pipeline for zh<->ms.",
        lifespan=lifespan,
    )
    app.mount("/outputs", StaticFiles(directory=settings.outputs_dir), name="outputs")

    @app.exception_handler(DemoError)
    async def demo_error_handler(_: Request, exc: DemoError):
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"code": exc.code, "message": exc.message, "detail": exc.detail}},
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(_: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "VALIDATION_ERROR",
                    "message": "Request validation failed.",
                    "detail": {"errors": jsonable_encoder(exc.errors())},
                }
            },
        )

    def current_pipeline(request: Request) -> SpeechTranslationPipeline:
        return request.app.state.pipeline

    async def read_upload(file: UploadFile) -> bytes:
        limit = settings.max_upload_mb * 1024 * 1024
        data = await file.read(limit + 1)
        if len(data) > limit:
            raise InvalidAudioError(
                f"Audio upload exceeds the {settings.max_upload_mb} MB limit."
            )
        return data

    @app.get("/health")
    async def health(request: Request):
        status = current_pipeline(request).status()
        model_ready = status["asr"].get("model_present", False) and all(
            model["present"] for model in status["translation"]["models"].values()
        )
        tts_ready = status["tts"].get("ready")
        if tts_ready is None:
            # Backward compatibility for simple/custom adapters.
            tts_ready = status["tts"].get("command_present", False) and all(
                voice.get("available", False)
                for voice in status["tts"].get("voices", {}).values()
            )
        return {
            "status": "ok" if model_ready and tts_ready else "degraded",
            "offline": settings.offline,
            "ready": model_ready and tts_ready,
            "components": status,
        }

    @app.post("/warmup")
    async def warmup(request: Request):
        return current_pipeline(request).warmup()

    @app.post("/asr")
    async def asr(
        request: Request,
        file: Annotated[UploadFile, File(...)],
        source_language: Annotated[str | None, Form()] = None,
        user_context: Annotated[str | None, Form()] = None,
    ):
        data = await read_upload(file)
        pipe = current_pipeline(request)
        audio, decode_ms = pipe.decode(data, file.filename)
        result = pipe.transcribe_audio(
            audio,
            source_language,
            user_context=user_context,
        )
        body = result.model_dump()
        body["raw_asr_text"] = result.raw_text
        body["enhanced_asr_text"] = result.text
        body["decode_ms"] = decode_ms
        return body

    @app.post("/translate")
    async def translate(request: Request, body: TranslationRequest):
        return current_pipeline(request).translate_text(
            body.text,
            body.source_language,
            body.target_language,
        )

    @app.post("/tts")
    async def tts(request: Request, body: TTSRequest):
        return current_pipeline(request).synthesize_text(
            body.text, body.language, voice_id=body.voice_id
        )

    @app.post("/edited-speech-translate")
    async def edited_speech_translate(
        request: Request, body: EditedSpeechRequest
    ):
        return current_pipeline(request).translate_edited_asr(
            request_id=body.request_id,
            parent_request_id=body.parent_request_id,
            raw_asr_text=body.raw_asr_text,
            edited_asr_text=body.edited_asr_text,
            source_language=body.source_language,
            target_language=body.target_language,
            voice_id=body.voice_id,
        )

    @app.post("/speech-translate")
    async def speech_translate(
        request: Request,
        file: Annotated[UploadFile, File(...)],
        source_language: Annotated[str | None, Form()] = None,
        target_language: Annotated[str | None, Form()] = None,
        voice_id: Annotated[str | None, Form()] = None,
        user_context: Annotated[str | None, Form()] = None,
        client_request_id: Annotated[str | None, Form()] = None,
    ):
        data = await read_upload(file)
        return current_pipeline(request).run(
            data,
            filename=file.filename,
            source_language=source_language,
            target_language=target_language,
            voice_id=voice_id,
            user_context=user_context,
            request_id=client_request_id,
        )

    web_dir = Path(__file__).resolve().parent / "web"
    if web_dir.is_dir():
        # Register last so API and OpenAPI routes retain precedence.
        app.mount("/", StaticFiles(directory=web_dir, html=True), name="web")

    return app


app = create_app()
