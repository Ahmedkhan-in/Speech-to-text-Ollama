import json
import os
import subprocess
import tempfile
from pathlib import Path

import requests
from faster_whisper import WhisperModel, decode_audio

from models import ConversationSummary, Segment, TranscriptionResult, Word

WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "small")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma3:4b")

_whisper: WhisperModel | None = None


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


def transcribe_audio(audio_path: str | Path, language: str | None = None) -> TranscriptionResult:
    converted_path = preprocess_audio(audio_path)
    try:
        model = _get_whisper()
        
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
            try:
                audio_data = decode_audio(str(converted_path), sampling_rate=16000)
                detected_lang, lang_prob, all_probs = model.detect_language(audio_data)
            except Exception:
                # Safe fallback if audio decoding fails.
                detected_lang, lang_prob, all_probs = "en", 1.0, [("en", 1.0)]
            
            probs_dict = {l: p for l, p in all_probs}
            en_prob = probs_dict.get("en", 0.0)
            ur_prob = probs_dict.get("ur", 0.0)
            
            # Double-check language selection: If Urdu has any non-trivial signal (>= 0.02)
            # or if the detected language is English with low confidence (< 0.85),
            # run BOTH English and Urdu transcriptions and select the higher confidence result.
            if (ur_prob >= 0.02) or (detected_lang == "en" and lang_prob < 0.85) or (detected_lang not in ("en", "ur")):
                # Urdu run
                try:
                    ur_segments_iter, _ur_info = model.transcribe(
                        str(converted_path),
                        language="ur",
                        initial_prompt=prompt_ur,
                        **transcribe_opts
                    )
                    ur_segments = list(ur_segments_iter)
                except Exception:
                    ur_segments = []
                
                # English run
                try:
                    en_segments_iter, _en_info = model.transcribe(
                        str(converted_path),
                        language="en",
                        initial_prompt=prompt_en,
                        **transcribe_opts
                    )
                    en_segments = list(en_segments_iter)
                except Exception:
                    en_segments = []
                
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
                else:
                    chosen_segments = en_segments
            else:
                # Highly confident language detection
                prompt = prompt_ur if detected_lang == "ur" else prompt_en
                segments_iter, _info = model.transcribe(
                    str(converted_path),
                    language=detected_lang,
                    initial_prompt=prompt,
                    **transcribe_opts
                )
                chosen_segments = list(segments_iter)
        else:
            # Explicit language override chosen by the user
            prompt = prompt_ur if language == "ur" else prompt_en
            segments_iter, _info = model.transcribe(
                str(converted_path),
                language=language,
                initial_prompt=prompt,
                **transcribe_opts
            )
            chosen_segments = list(segments_iter)
        
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


def _call_ollama(prompt: str, json_format: bool = False) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
    }
    if json_format:
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


def extract_summary(transcript: str) -> ConversationSummary:
    last_error: Exception | None = None
    for strict in (False, True):
        try:
            raw = _call_ollama(_build_prompt(transcript, strict=strict), json_format=True)
            return ConversationSummary.model_validate_json(raw)
        except Exception as exc:
            last_error = exc
    raise ValueError(f"Could not parse structured summary: {last_error}")


def save_audio_bytes(data: bytes, suffix: str) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(data)
        return tmp.name


def run_pipeline(audio_path: str | Path, language: str | None = None) -> tuple[TranscriptionResult, ConversationSummary]:
    transcription = transcribe_audio(audio_path, language=language)
    summary = extract_summary(transcription.text)
    return transcription, summary

