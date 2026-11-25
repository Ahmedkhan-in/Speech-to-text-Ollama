import base64
import html
import os
import queue
import threading
import time
from pathlib import Path
import streamlit as st
import streamlit.components.v1 as components

from pipeline import OLLAMA_MODEL, extract_summary, save_audio_bytes, transcribe_audio
from models import ConversationSummary, TranscriptionResult

st.set_page_config(page_title="Voice Transcription", layout="centered")

st.markdown(
    """
    <style>
    .block-container { padding-top: 2.5rem; max-width: 720px; }
    h1 { font-weight: 600; letter-spacing: -0.02em; margin-bottom: 0.25rem; }
    .subtitle { color: #6b7280; font-size: 1.05rem; margin-bottom: 2rem; }
    .badge {
        display: inline-block; padding: 0.2rem 0.65rem; margin-right: 0.5rem;
        border-radius: 999px; font-size: 0.8rem; font-weight: 500;
        background: #f3f4f6; color: #374151;
    }
    .section-label {
        font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.08em;
        color: #9ca3af; margin-bottom: 0.5rem;
    }
    .step { margin: 0.15rem 0; line-height: 1.5; }
    .step-pending { color: #9ca3af; }
    .step-time {
        font-family: ui-monospace, monospace; font-size: 0.8rem; color: #6b7280;
        margin-left: 0.35rem;
    }
    .step-detail { color: #6b7280; }
    .step-note { color: #6b7280; font-size: 0.88rem; margin: 0.1rem 0 0.1rem 1.6rem; }
    .live-preview {
        border-left: 3px solid #e5e7eb; padding: 0.25rem 0 0.25rem 0.75rem;
        color: #4b5563; font-style: italic; font-size: 0.92rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Voice Transcription")
st.markdown(
    '<p class="subtitle">Transcribe conversations and extract structured insights locally.</p>',
    unsafe_allow_html=True,
)

language_option = st.selectbox(
    "Select Language",
    options=["Auto Detect", "English", "Urdu"],
    index=0,
    help="Explicitly override the speech language or let Whisper auto-detect."
)
language_map = {
    "Auto Detect": None,
    "English": "en",
    "Urdu": "ur"
}
selected_language = language_map[language_option]


def audio_recorder(key=None):
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    component_dir = os.path.join(parent_dir, "audio_recorder_component")
    _component_func = components.declare_component("audio_recorder", path=component_dir)
    return _component_func(key=key)


record_tab, upload_tab = st.tabs(["Record", "Upload"])
audio_bytes: bytes | None = None
audio_suffix = ".wav"

with record_tab:
    audio_data = audio_recorder(key="custom_audio_recorder")

with upload_tab:
    uploaded = st.file_uploader("Upload audio", type=["wav", "mp3", "m4a"])

if uploaded is not None:
    audio_bytes = uploaded.getvalue()
    audio_suffix = Path(uploaded.name).suffix or ".wav"
else:
    if audio_data and audio_data.get("status") == "stopped":
        audio_bytes = base64.b64decode(audio_data["audio_base64"])
        mime_type = audio_data.get("audio_type", "")
        # ponytail: extract proper extension from mime type
        if "mp4" in mime_type:
            audio_suffix = ".mp4"
        elif "ogg" in mime_type:
            audio_suffix = ".ogg"
        elif "wav" in mime_type:
            audio_suffix = ".wav"
        else:
            audio_suffix = ".webm"




def format_timestamp(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes:02d}:{secs:02d}"


def render_segments(transcription: TranscriptionResult) -> None:
    lines = [
        f"[{format_timestamp(s.start)} → {format_timestamp(s.end)}] {s.text}"
        for s in transcription.segments
    ]
    st.text("\n".join(lines) if lines else transcription.text)


def render_summary(summary: ConversationSummary) -> None:
    st.markdown(f"## {summary.title}")
    st.markdown(
        f'<span class="badge">{summary.conversation_type.title()}</span>'
        f'<span class="badge">{summary.sentiment.title()}</span>',
        unsafe_allow_html=True,
    )
    if summary.duration_estimate_minutes is not None:
        st.caption(f"~{summary.duration_estimate_minutes:.1f} min estimated")
    st.markdown(summary.summary)
    if summary.participants:
        st.markdown('<p class="section-label">Participants</p>', unsafe_allow_html=True)
        st.write(", ".join(summary.participants))
    if summary.key_points:
        st.markdown('<p class="section-label">Key points</p>', unsafe_allow_html=True)
        for point in summary.key_points:
            st.markdown(f"- {point}")
    if summary.action_items:
        st.markdown('<p class="section-label">Action items</p>', unsafe_allow_html=True)
        for item in summary.action_items:
            st.markdown(f"- [ ] {item}")


# ---------------------------------------------------------------------------
# Live progress: the pipeline runs in a background thread and reports events;
# this script redraws the step list, timer and live transcript a few times a second.
# ---------------------------------------------------------------------------

# Stages shown up front (greyed out until they start). "detect" only appears in Auto Detect.
STAGE_PLAN = [
    ("preprocess", "Clean audio"),
    ("model", "Load Whisper model"),
    ("detect", "Detect language"),
    ("transcribe", "Transcribe"),
    ("summary", "Extract insights"),
]
# Short names for the status header, by step-key prefix.
PHASE_NAMES = {
    "preprocess": "Cleaning audio",
    "model": "Loading model",
    "detect": "Detecting language",
    "transcribe": "Transcribing",
    "summary": "Extracting insights",
}
REFRESH_SECONDS = 0.2


def clock(seconds: float) -> str:
    """Running timer format, e.g. 01:24.6"""
    minutes, secs = divmod(seconds, 60)
    return f"{int(minutes):02d}:{secs:04.1f}"


def human_duration(seconds: float) -> str:
    """Compact duration, e.g. 4.2s or 2m 41s"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(int(round(seconds)), 60)
    return f"{minutes}m {secs:02d}s"


class RunTracker:
    """Turns pipeline progress events into the state the live status box displays."""

    def __init__(self, auto_detect: bool):
        self.started = time.perf_counter()
        self.steps: dict[str, dict] = {}
        self.order: list[str] = []
        self.phase = "Starting"
        self.progress: tuple[float, str] | None = None
        self.preview: list[str] = []
        self.audio_seconds: float | None = None
        self._notes = 0
        for key, label in STAGE_PLAN:
            if key == "detect" and not auto_detect:
                continue
            self._add(key, label)

    def _add(self, key: str, label: str, index: int | None = None, kind: str = "step") -> None:
        self.steps[key] = {"label": label, "status": "pending", "kind": kind,
                           "started": None, "ended": None, "detail": ""}
        self.order.insert(len(self.order) if index is None else index, key)

    def _first_pending_index(self) -> int:
        for i, key in enumerate(self.order):
            if self.steps[key]["status"] == "pending":
                return i
        return len(self.order)

    def handle(self, event: str, data: dict, at: float) -> None:
        if event == "step_start":
            key = data["key"]
            if key not in self.steps:
                base = key.split("_")[0]
                if base in self.steps and self.steps[base]["status"] == "pending":
                    # Replace the generic placeholder (e.g. "transcribe" -> "transcribe_ur").
                    index = self.order.index(base)
                    self.order.pop(index)
                    del self.steps[base]
                else:
                    index = self._first_pending_index()
                self._add(key, data["label"], index)
            step = self.steps[key]
            step.update(label=data["label"], status="running", started=at)
            self.phase = PHASE_NAMES.get(key.split("_")[0], data["label"])
            if key.startswith("transcribe"):
                self.preview = []
        elif event == "step_done":
            step = self.steps.get(data["key"])
            if step:
                step.update(status="done", ended=at, detail=data.get("detail", ""))
                if step["started"] is None:
                    step["started"] = at
            if data["key"].startswith("transcribe"):
                self.progress = None
                self.preview = []
        elif event == "segment":
            duration = data.get("duration") or 0
            self.audio_seconds = duration or self.audio_seconds
            done = min(data["end"] / duration, 1.0) if duration else 0.0
            overall = ((data["pass_index"] - 1) + done) / data["pass_count"]
            pass_text = (f"Pass {data['pass_index']} of {data['pass_count']}: "
                         if data["pass_count"] > 1 else "")
            self.progress = (overall, f"{pass_text}{done:.0%} · {format_timestamp(data['end'])} "
                                      f"of {format_timestamp(duration)} audio")
            if data["text"]:
                self.preview = (self.preview + [data["text"]])[-3:]
        elif event == "info":
            self._notes += 1
            key = f"note_{self._notes}"
            self._add(key, data["message"], self._first_pending_index(), kind="note")
            self.steps[key]["status"] = "done"

    def fail_running(self, at: float) -> None:
        for step in self.steps.values():
            if step["status"] == "running":
                step.update(status="failed", ended=at)

    def step_time(self, prefix: str) -> float:
        return sum(
            s["ended"] - s["started"]
            for k, s in self.steps.items()
            if k.startswith(prefix) and s["kind"] == "step" and s["started"] and s["ended"]
        )

    def render_steps(self, now: float) -> str:
        rows = []
        for key in self.order:
            s = self.steps[key]
            label = html.escape(s["label"])
            if s["kind"] == "note":
                rows.append(f'<div class="step-note">ℹ️ {label}</div>')
                continue
            if s["status"] == "done":
                detail = f' <span class="step-detail">· {html.escape(s["detail"])}</span>' if s["detail"] else ""
                took = human_duration(s["ended"] - s["started"])
                rows.append(f'<div class="step">✅ {label}{detail}<span class="step-time">{took}</span></div>')
            elif s["status"] == "running":
                took = human_duration(now - s["started"])
                rows.append(f'<div class="step">⏳ <b>{label}</b><span class="step-time">{took}</span></div>')
            elif s["status"] == "failed":
                rows.append(f'<div class="step">❌ {label}</div>')
            else:
                rows.append(f'<div class="step step-pending">○ {label}</div>')
        return "\n".join(rows)


def run_with_live_status(audio_path: str, language: str | None):
    """Run transcription + summary in a worker thread while showing live progress and a timer."""
    events: queue.Queue = queue.Queue()
    result: dict = {}

    def on_progress(event: str, data: dict) -> None:
        events.put((event, data, time.perf_counter()))

    def worker() -> None:
        try:
            transcription = transcribe_audio(audio_path, language=language, on_progress=on_progress)
            result["transcription"] = transcription
            result["summary"] = extract_summary(transcription.text, on_progress=on_progress)
        except Exception as exc:  # surfaced in the main script below
            result["error"] = exc

    tracker = RunTracker(auto_detect=language is None)
    status = st.status(f"Starting… ⏱ {clock(0)}", expanded=True)
    with status:
        steps_box = st.empty()
        progress_box = st.empty()
        preview_box = st.empty()

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    while True:
        finished = not thread.is_alive()  # check before draining so no final event is missed
        while True:
            try:
                tracker.handle(*events.get_nowait())
            except queue.Empty:
                break
        now = time.perf_counter()
        if finished:
            break
        # state must be passed every time, or Streamlit drops the spinner icon.
        status.update(label=f"{tracker.phase}… ⏱ {clock(now - tracker.started)}", state="running")
        steps_box.markdown(tracker.render_steps(now), unsafe_allow_html=True)
        if tracker.progress:
            value, text = tracker.progress
            progress_box.progress(value, text=text)
        else:
            progress_box.empty()
        if tracker.preview:
            lines = "<br>".join(html.escape(line) for line in tracker.preview)
            preview_box.markdown(f'<div class="live-preview">{lines}</div>', unsafe_allow_html=True)
        else:
            preview_box.empty()
        time.sleep(REFRESH_SECONDS)

    thread.join()
    total = time.perf_counter() - tracker.started
    progress_box.empty()
    preview_box.empty()
    if "error" in result:
        tracker.fail_running(now)
        steps_box.markdown(tracker.render_steps(now), unsafe_allow_html=True)
        status.update(label=f"Failed after {human_duration(total)}", state="error", expanded=True)
        raise result["error"]
    steps_box.markdown(tracker.render_steps(now), unsafe_allow_html=True)
    status.update(label=f"Done in {human_duration(total)} · click to see what happened",
                  state="complete", expanded=False)
    return result["transcription"], result["summary"], tracker, total


def render_timing(tracker: RunTracker, total: float) -> None:
    prep = tracker.step_time("preprocess") + tracker.step_time("model") + tracker.step_time("detect")
    transcribe_time = tracker.step_time("transcribe")
    cols = st.columns(4)
    cols[0].metric("Total time", human_duration(total))
    cols[1].metric("Audio prep", human_duration(prep),
                   help="FFmpeg cleanup, loading Whisper and language detection")
    cols[2].metric("Transcription", human_duration(transcribe_time))
    cols[3].metric("Insights", human_duration(tracker.step_time("summary")),
                   help=f"Summary by {OLLAMA_MODEL} via Ollama")
    if tracker.audio_seconds and transcribe_time:
        ratio = transcribe_time / tracker.audio_seconds
        st.caption(f"Audio length {format_timestamp(tracker.audio_seconds)} · "
                   f"transcription took {ratio:.1f}× the audio length")


if st.button("Transcribe and Analyze", type="primary", use_container_width=True):
    if not audio_bytes:
        st.error("Please record or upload an audio file first.")
    else:
        temp_path: str | None = None
        try:
            temp_path = save_audio_bytes(audio_bytes, audio_suffix)
            transcription, summary, tracker, total = run_with_live_status(temp_path, selected_language)

            st.divider()
            render_timing(tracker, total)
            st.subheader("Results")
            with st.expander("Transcript", expanded=True):
                st.write(transcription.text)
                if transcription.segments:
                    st.markdown("**Segments**")
                    render_segments(transcription)
            render_summary(summary)
            with st.expander("View raw JSON"):
                st.json(summary.model_dump())
        except Exception as exc:
            st.error(str(exc))
        finally:
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)
