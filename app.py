import os
from pathlib import Path

import streamlit as st

from pipeline import extract_summary, save_audio_bytes, transcribe_audio
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
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Voice Transcription")
st.markdown(
    '<p class="subtitle">Transcribe conversations and extract structured insights locally.</p>',
    unsafe_allow_html=True,
)

record_tab, upload_tab = st.tabs(["Record", "Upload"])
audio_bytes: bytes | None = None
audio_suffix = ".wav"

with record_tab:
    recording = st.audio_input("Record a conversation")
    if recording is not None:
        audio_bytes = recording.getvalue()
        audio_suffix = ".wav"

with upload_tab:
    uploaded = st.file_uploader("Upload audio", type=["wav", "mp3", "m4a"])
    if uploaded is not None:
        audio_bytes = uploaded.getvalue()
        audio_suffix = Path(uploaded.name).suffix or ".wav"


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


if st.button("Transcribe and Analyze", type="primary", use_container_width=True):
    if not audio_bytes:
        st.error("Please record or upload an audio file first.")
    else:
        temp_path: str | None = None
        try:
            temp_path = save_audio_bytes(audio_bytes, audio_suffix)
            with st.spinner("Transcribing audio..."):
                transcription = transcribe_audio(temp_path)
            with st.spinner("Extracting structured insights..."):
                summary = extract_summary(transcription.text)

            st.divider()
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
