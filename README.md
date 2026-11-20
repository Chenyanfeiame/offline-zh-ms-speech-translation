# Offline Chinese–Malay Speech Translation

An end-to-end, fully offline Mandarin Chinese ↔ Malay speech translation demo for Apple Silicon.

The production path is:

`Microphone → language ID → Qwen3-ASR → conservative entity rescue → editable transcript → HY-MT → semantic number verbalization → Piper/MMS TTS`

## What is demonstrated

- Automatic Chinese/Malay speech-language detection with a local Core ML model.
- One shared 4-bit Qwen3-ASR model routed for both languages.
- Request-scoped phonetic entity retrieval for place names, food names and other low-frequency terms. There is no business-scenario classifier and no global thousand-word prompt.
- Raw and enhanced transcripts remain separate in the backend audit trail; only conservative, reproducible corrections are accepted.
- The transcript is editable in the UI before MT/TTS, and stale responses are rejected by `request_id`.
- Stateless HY-MT requests with no shared conversation history, session ID or prompt cache.
- A bilingual semantic-number layer keeps display text unchanged while producing a separate TTS text: phone/order/flight identifiers are read digit by digit; money, duration, dates and quantities use normal number readings.
- Offline Piper Chinese TTS and MMS Malay TTS.

## Measured ASR results

Evaluation used fixed reference text and kept Chinese and Malay FLEURS subsets separate.

| Set | Final ASR | Metric | Result |
|---|---|---:|---:|
| Chinese FLEURS (50 utterances) | Qwen3-ASR 1.7B 4-bit | CER | 7.62% |
| Malay FLEURS (50 utterances) | Qwen3-ASR 1.7B 4-bit | WER | 11.63% |

The conservative Chinese entity layer made zero changes on the 50-utterance FLEURS set, so CER remained 7.62% with 0 false corrections. This is intentional: the layer activates only when phonetic evidence is strong enough. Model weights, FLEURS data, personal recordings and generated reports are not committed to this repository.

## Requirements

- Apple Silicon Mac
- Python 3.10–3.12
- `ffmpeg`
- `llama.cpp` (`brew install llama.cpp`)
- Enough local disk space for the pinned ASR, language-ID, MT and TTS models

## Setup

```bash
./scripts/bootstrap.sh
source .venv/bin/activate
python scripts/prepare_models.py
cp config.example.json config.json
zhms doctor
zhms serve --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000>. The first model load is slower; subsequent requests reuse the warmed local models.

The model downloader pins exact upstream revisions and writes `models/manifest.json` with sizes and SHA-256 hashes. Review each upstream model license before redistribution; the application source itself is MIT licensed.

## Test

```bash
pytest
python scripts/test_mt_request_isolation.py
```

The unit suite covers request isolation, Qwen routing, entity rescue, number semantics, translation payloads, editable-text/TTS behavior and the recording UI contract. The MT isolation script runs sequential and concurrent A/B/C cases against the real local model and writes request-level audit logs under `reports/`.

## Repository scope

This public repository contains only the final application, entity lexicon, reproducibility scripts and tests. It intentionally excludes model weights, audio, generated WAV files, local configuration, evaluation corpora, old experiments, backups and private reports.
