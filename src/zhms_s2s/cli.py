from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import Settings
from .runtime import build_pipeline


def _print(value) -> None:
    if hasattr(value, "model_dump"):
        value = value.model_dump()
    print(json.dumps(value, ensure_ascii=False, indent=2))


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="zhms", description="Offline zh<->ms speech translation")
    parser.add_argument("--config", default=None, help="Path to JSON config (default: config.json)")
    sub = parser.add_subparsers(dest="command", required=True)

    doctor = sub.add_parser("doctor", help="Inspect model and TTS readiness without loading models")
    doctor.set_defaults(action="doctor")

    translate = sub.add_parser("translate", help="Translate text")
    translate.add_argument("--source", required=True, choices=["zh", "ms"])
    translate.add_argument("--target", choices=["zh", "ms"])
    translate.add_argument("text")

    tts = sub.add_parser("tts", help="Synthesize local speech")
    tts.add_argument("--language", required=True, choices=["zh", "ms"])
    tts.add_argument("--voice")
    tts.add_argument("text")

    asr = sub.add_parser("asr", help="Transcribe an audio file")
    asr.add_argument("audio", type=Path)
    asr.add_argument("--source", choices=["zh", "ms"])
    asr.add_argument("--context", help="Comma-separated known names, codes, or dates")

    speech = sub.add_parser("speech-translate", help="Run ASR -> MT -> TTS")
    speech.add_argument("audio", type=Path)
    speech.add_argument("--source", choices=["zh", "ms"])
    speech.add_argument("--target", choices=["zh", "ms"])
    speech.add_argument("--voice")
    speech.add_argument("--context", help="Comma-separated known names, codes, or dates")

    serve = sub.add_parser("serve", help="Run FastAPI service")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    return parser


def main() -> None:
    args = make_parser().parse_args()
    settings = Settings.load(args.config)
    if args.command == "serve":
        import os
        import uvicorn

        if args.config:
            os.environ["ZHMS_CONFIG"] = str(Path(args.config).resolve())
        uvicorn.run("zhms_s2s.api:app", host=args.host, port=args.port, reload=False)
        return

    pipeline = build_pipeline(settings)
    if args.command == "doctor":
        _print(pipeline.status())
    elif args.command == "translate":
        _print(
            pipeline.translate_text(
                args.text,
                args.source,
                args.target,
            )
        )
    elif args.command == "tts":
        _print(pipeline.synthesize_text(args.text, args.language, voice_id=args.voice))
    elif args.command == "asr":
        data = args.audio.read_bytes()
        audio, decode_ms = pipeline.decode(data, args.audio.name)
        result = pipeline.transcribe_audio(
            audio,
            args.source,
            user_context=args.context,
        ).model_dump()
        result["decode_ms"] = decode_ms
        _print(result)
    elif args.command == "speech-translate":
        _print(
            pipeline.run(
                args.audio.read_bytes(),
                filename=args.audio.name,
                source_language=args.source,
                target_language=args.target,
                voice_id=args.voice,
                user_context=args.context,
            )
        )


if __name__ == "__main__":
    main()
