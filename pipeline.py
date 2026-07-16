import json
import os
import tempfile
from pathlib import Path

import requests
from faster_whisper import WhisperModel

from models import ConversationSummary, Segment, TranscriptionResult

WHISPER_MODEL = "small"
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma3:4b")

_whisper: WhisperModel | None = None


def _get_whisper() -> WhisperModel:
    global _whisper
    if _whisper is None:
        _whisper = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
    return _whisper


def transcribe_audio(audio_path: str | Path) -> TranscriptionResult:
    segments_iter, _info = _get_whisper().transcribe(str(audio_path), beam_size=5)
    segments: list[Segment] = []
    parts: list[str] = []
    for seg in segments_iter:
        text = seg.text.strip()
        if not text:
            continue
        segments.append(Segment(start=seg.start, end=seg.end, text=text))
        parts.append(text)
    return TranscriptionResult(text=" ".join(parts), segments=segments)


def _build_prompt(transcript: str, strict: bool = False) -> str:
    schema = json.dumps(ConversationSummary.model_json_schema(), indent=2)
    prefix = (
        "Return ONLY valid JSON matching this schema. No markdown, no explanation.\n\n"
        if strict
        else "Analyze the conversation transcript and return JSON matching this schema.\n\n"
    )
    return (
        f"{prefix}Schema:\n{schema}\n\n"
        f"Transcript:\n{transcript}\n"
    )


def _call_ollama(prompt: str) -> str:
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "format": "json",
        },
        timeout=180,
    )
    response.raise_for_status()
    return response.json()["response"]


def extract_summary(transcript: str) -> ConversationSummary:
    last_error: Exception | None = None
    for strict in (False, True):
        try:
            raw = _call_ollama(_build_prompt(transcript, strict=strict))
            return ConversationSummary.model_validate_json(raw)
        except Exception as exc:
            last_error = exc
    raise ValueError(f"Could not parse structured summary: {last_error}")


def save_audio_bytes(data: bytes, suffix: str) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(data)
        return tmp.name


def run_pipeline(audio_path: str | Path) -> tuple[TranscriptionResult, ConversationSummary]:
    transcription = transcribe_audio(audio_path)
    summary = extract_summary(transcription.text)
    return transcription, summary
