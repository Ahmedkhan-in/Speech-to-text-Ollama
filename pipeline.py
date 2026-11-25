import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Callable

import requests
from faster_whisper import WhisperModel, decode_audio

from models import ConversationSummary, Segment, TranscriptionResult, Word

WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "small")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma3:4b")
OLLAMA_KEEP_ALIVE = os.environ.get("OLLAMA_KEEP_ALIVE", "30m")

_whisper: WhisperModel | None = None

# Optional progress reporting. Callers pass on_progress(event, data) to watch the
# pipeline work; events are:
#   "step_start" {key, label}            a stage began
#   "step_done"  {key, detail}           a stage finished (detail may be "")
#   "segment"    {key, text, end, duration, pass_index, pass_count}
#                                        one transcript segment was decoded
#   "info"       {message}               a notable decision (e.g. dual-language run)
ProgressCallback = Callable[[str, dict[str, Any]], None]

LANGUAGE_NAMES = {"en": "English", "ur": "Urdu"}


def _emit(on_progress: ProgressCallback | None, event: str, **data: Any) -> None:
    if on_progress is None:
        return
    try:
        on_progress(event, data)
    except Exception:
        # Progress reporting must never break transcription.
        pass


def _format_duration(seconds: float) -> str:
    minutes, secs = divmod(int(round(seconds)), 60)
    return f"{minutes}:{secs:02d}"


def _get_whisper() -> WhisperModel:
    global _whisper
    if _whisper is None:
        # Resolve model name from configuration variable and optimize for CPU
        # using int8 quantization and 4 threads matching physical cores.
        _whisper = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8", cpu_threads=4)
        
    return _whisper


def preprocess_audio(input_path: str | Path) -> str | Path:
    # ponytail: WAV, PCM 16-bit, Mono, 16 kHz conversion via FFmpeg
    output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
    
    # Try advanced filters: highpass at 80Hz to remove mic rumble, lowpass at 8000Hz to remove hiss,
    # and speechnorm for speech-specific loudness normalization.
    cmd_advanced = [
        "ffmpeg", "-y", "-i", str(input_path),
        "-af", "highpass=f=80,lowpass=f=8000,speechnorm",
        "-acodec", "pcm_s16le", "-ac", "1", "-ar", "16000",
        output_path
    ]
    
    # Fallback 1: Use dynaudnorm instead of speechnorm if speechnorm filter is not supported.
    cmd_dynaudnorm = [
        "ffmpeg", "-y", "-i", str(input_path),
        "-af", "highpass=f=80,lowpass=f=8000,dynaudnorm",
        "-acodec", "pcm_s16le", "-ac", "1", "-ar", "16000",
        output_path
    ]
    
    # Fallback 2: Basic format conversion if all advanced audio filtering fails.
    cmd_basic = [
        "ffmpeg", "-y", "-i", str(input_path),
        "-acodec", "pcm_s16le", "-ac", "1", "-ar", "16000",
        output_path
    ]
    
    for cmd in (cmd_advanced, cmd_dynaudnorm, cmd_basic):
        try:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            return output_path
        except Exception:
            if os.path.exists(output_path):
                try:
                    os.unlink(output_path)
                except Exception:
                    pass
    
    return input_path


def _run_pass(
    model: WhisperModel,
    audio_path: str | Path,
    language: str,
    prompt: str,
    opts: dict,
    on_progress: ProgressCallback | None,
    key: str,
    label: str,
    pass_index: int = 1,
    pass_count: int = 1,
) -> list:
    """Run one Whisper pass, consuming segments as they are decoded so progress can be reported."""
    _emit(on_progress, "step_start", key=key, label=label)
    segments_iter, info = model.transcribe(
        str(audio_path), language=language, initial_prompt=prompt, **opts
    )
    segments = []
    for seg in segments_iter:
        segments.append(seg)
        _emit(
            on_progress, "segment",
            key=key, text=seg.text.strip(), end=seg.end, duration=info.duration,
            pass_index=pass_index, pass_count=pass_count,
        )
    _emit(on_progress, "step_done", key=key, detail=f"{len(segments)} segments")
    return segments


def transcribe_audio(
    audio_path: str | Path,
    language: str | None = None,
    on_progress: ProgressCallback | None = None,
) -> TranscriptionResult:
    _emit(on_progress, "step_start", key="preprocess",
          label="Cleaning audio with FFmpeg (mono 16 kHz, rumble/hiss filters, loudness)")
    converted_path = preprocess_audio(audio_path)
    if converted_path == audio_path:
        _emit(on_progress, "step_done", key="preprocess", detail="FFmpeg failed, using original audio")
    else:
        audio_seconds = max(os.path.getsize(converted_path) - 44, 0) / 32000  # 16 kHz * 16-bit mono
        _emit(on_progress, "step_done", key="preprocess",
              detail=f"{_format_duration(audio_seconds)} of audio")
    try:
        already_loaded = _whisper is not None
        _emit(on_progress, "step_start", key="model", label=f"Loading Whisper '{WHISPER_MODEL}' model")
        model = _get_whisper()
        _emit(on_progress, "step_done", key="model",
              detail="already in memory" if already_loaded else "loaded")

        # Optimize transcription parameters for maximum accuracy:
        # - beam_size=5: recommended search depth balancing accuracy and latency.
        # - temperature=0.0: deterministic greedy decoding for consistency.
        # - condition_on_previous_text=False: CRITICAL to prevent error propagation and repetition loops.
        # - vad_filter=True: enables voice activity detection.
        # - vad_parameters: customized to group speech cleanly and handle silence.
        transcribe_opts = {
            "beam_size": 5,
            "temperature": 0.0,
            "condition_on_previous_text": False,
            "vad_filter": True,
            "vad_parameters": {
                "threshold": 0.5,
                "min_speech_duration_ms": 250,
                "min_silence_duration_ms": 1000,
                "speech_pad_ms": 400,
            },
            "word_timestamps": True,
        }
        
        # Initial prompts to establish vocabulary, grammar and proper punctuation.
        prompt_en = "This is a clean, punctuated transcript of a conversation."
        prompt_ur = "یہ ایک صاف ستھری گفتگو کی ریکارڈنگ ہے جس میں اردو زبان استعمال کی گئی ہے۔"
        
        chosen_segments = []
        
        if language is None:
            # Auto-detect mode. Optimize Urdu detection specifically to combat English bias.
            _emit(on_progress, "step_start", key="detect", label="Detecting spoken language")
            try:
                audio_data = decode_audio(str(converted_path), sampling_rate=16000)
                detected_lang, lang_prob, all_probs = model.detect_language(audio_data)
                _emit(on_progress, "step_done", key="detect",
                      detail=f"{LANGUAGE_NAMES.get(detected_lang, detected_lang)} ({lang_prob:.0%} confident)")
            except Exception:
                # Safe fallback if audio decoding fails.
                detected_lang, lang_prob, all_probs = "en", 1.0, [("en", 1.0)]
                _emit(on_progress, "step_done", key="detect", detail="detection failed, assuming English")

            probs_dict = {l: p for l, p in all_probs}
            en_prob = probs_dict.get("en", 0.0)
            ur_prob = probs_dict.get("ur", 0.0)
            
            # Double-check language selection: If Urdu has any non-trivial signal (>= 0.02)
            # or if the detected language is English with low confidence (< 0.85),
            # run BOTH English and Urdu transcriptions and select the higher confidence result.
            if (ur_prob >= 0.02) or (detected_lang == "en" and lang_prob < 0.85) or (detected_lang not in ("en", "ur")):
                _emit(on_progress, "info",
                      message="Language is uncertain, so transcribing in both Urdu and English "
                              "and keeping the more confident result.")
                # Urdu run
                try:
                    ur_segments = _run_pass(
                        model, converted_path, "ur", prompt_ur, transcribe_opts, on_progress,
                        key="transcribe_ur", label="Transcribing, pass 1 of 2 (Urdu)",
                        pass_index=1, pass_count=2,
                    )
                except Exception:
                    ur_segments = []
                    _emit(on_progress, "step_done", key="transcribe_ur", detail="failed, skipped")

                # English run
                try:
                    en_segments = _run_pass(
                        model, converted_path, "en", prompt_en, transcribe_opts, on_progress,
                        key="transcribe_en", label="Transcribing, pass 2 of 2 (English)",
                        pass_index=2, pass_count=2,
                    )
                except Exception:
                    en_segments = []
                    _emit(on_progress, "step_done", key="transcribe_en", detail="failed, skipped")
                
                # Scoring function based on average log probability weighted by segment duration
                def get_score(segs):
                    if not segs:
                        return -999.0
                    total_dur = 0.0
                    sum_logprob = 0.0
                    for seg in segs:
                        dur = max(seg.end - seg.start, 0.1)
                        sum_logprob += seg.avg_logprob * dur
                        total_dur += dur
                    return (sum_logprob / total_dur) if total_dur > 0 else -999.0
                
                ur_score = get_score(ur_segments)
                en_score = get_score(en_segments)
                
                if ur_score > en_score and ur_score > -999.0:
                    chosen_segments = ur_segments
                    kept = "Urdu"
                else:
                    chosen_segments = en_segments
                    kept = "English"
                _emit(on_progress, "info",
                      message=f"Kept the {kept} transcript (confidence Urdu {ur_score:.2f} "
                              f"vs English {en_score:.2f}, higher is better).")
            else:
                # Highly confident language detection
                prompt = prompt_ur if detected_lang == "ur" else prompt_en
                chosen_segments = _run_pass(
                    model, converted_path, detected_lang, prompt, transcribe_opts, on_progress,
                    key="transcribe",
                    label=f"Transcribing ({LANGUAGE_NAMES.get(detected_lang, detected_lang)})",
                )
        else:
            # Explicit language override chosen by the user
            prompt = prompt_ur if language == "ur" else prompt_en
            chosen_segments = _run_pass(
                model, converted_path, language, prompt, transcribe_opts, on_progress,
                key="transcribe", label=f"Transcribing ({LANGUAGE_NAMES.get(language, language)})",
            )
        
        segments: list[Segment] = []
        parts: list[str] = []
        for seg in chosen_segments:
            text = seg.text.strip()
            if not text:
                continue
            
            words_list = []
            if seg.words:
                for w in seg.words:
                    words_list.append(Word(word=w.word, start=w.start, end=w.end, probability=w.probability))
            
            segments.append(Segment(
                start=seg.start,
                end=seg.end,
                text=text,
                words=words_list if words_list else None
            ))
            parts.append(text)
            
        raw_text = " ".join(parts)
        # Note: We return the raw Whisper output directly to prevent LLM rewrite hallucinations,
        # ensuring absolute fidelity to the spoken words and boosting performance.
        return TranscriptionResult(text=raw_text, segments=segments)
    finally:
        if converted_path != audio_path and os.path.exists(converted_path):
            try:
                os.unlink(converted_path)
            except Exception:
                pass


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


def _call_ollama(prompt: str, json_format: bool = False, schema: dict | None = None) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        # Keep the model in memory between requests (Ollama's default is 5 minutes),
        # so consecutive runs skip reloading it from disk.
        "keep_alive": OLLAMA_KEEP_ALIVE,
    }
    if schema is not None:
        # Structured output: Ollama constrains generation to this JSON schema,
        # so every required field is filled on the first attempt.
        payload["format"] = schema
    elif json_format:
        payload["format"] = "json"
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json=payload,
        timeout=180,
    )
    response.raise_for_status()
    return response.json()["response"]


def cleanup_transcript(transcript: str) -> str:
    if not transcript.strip():
        return transcript
    prompt = (
        "Restore punctuation, capitalization, and fix obvious spelling mistakes "
        "in the following transcript. Do NOT summarize, do NOT rephrase, do NOT translate, "
        "do NOT invent missing words, and preserve the original wording as closely as possible. "
        "Return ONLY the cleaned transcript without any explanation, prefix, or markdown formatting.\n\n"
        f"Transcript:\n{transcript}"
    )
    try:
        response = _call_ollama(prompt, json_format=False)
        cleaned = response.strip()
        if cleaned:
            return cleaned
    except Exception:
        pass
    return transcript


def extract_summary(transcript: str, on_progress: ProgressCallback | None = None) -> ConversationSummary:
    _emit(on_progress, "step_start", key="summary",
          label=f"Extracting insights with {OLLAMA_MODEL} (summary, key points, action items)")
    # First attempt: schema-constrained output (Ollama structured outputs). Without the
    # schema, gemma3 in plain JSON mode often replies with an empty "{}".
    # Fallback: the older strict-prompt JSON mode, in case the server lacks schema support.
    attempts = [
        {"strict": False, "schema": ConversationSummary.model_json_schema()},
        {"strict": True, "schema": None},
    ]
    last_error: Exception | None = None
    for i, attempt in enumerate(attempts):
        try:
            raw = _call_ollama(
                _build_prompt(transcript, strict=attempt["strict"]),
                json_format=True,
                schema=attempt["schema"],
            )
            summary = ConversationSummary.model_validate_json(raw)
            _emit(on_progress, "step_done", key="summary",
                  detail=f"{len(summary.key_points)} key points, {len(summary.action_items)} action items")
            return summary
        except Exception as exc:
            last_error = exc
            if i < len(attempts) - 1:
                _emit(on_progress, "info",
                      message="The model's reply didn't match the expected format, retrying with a stricter prompt.")
    raise ValueError(f"Could not parse structured summary: {last_error}")


def save_audio_bytes(data: bytes, suffix: str) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(data)
        return tmp.name


def run_pipeline(
    audio_path: str | Path,
    language: str | None = None,
    on_progress: ProgressCallback | None = None,
) -> tuple[TranscriptionResult, ConversationSummary]:
    transcription = transcribe_audio(audio_path, language=language, on_progress=on_progress)
    summary = extract_summary(transcription.text, on_progress=on_progress)
    return transcription, summary

