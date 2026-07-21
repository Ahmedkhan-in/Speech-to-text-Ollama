from typing import Literal, Optional

from pydantic import BaseModel, Field


class Word(BaseModel):
    word: str
    start: float
    end: float
    probability: float


class Segment(BaseModel):
    start: float
    end: float
    text: str
    words: Optional[list[Word]] = None


class TranscriptionResult(BaseModel):
    text: str
    segments: list[Segment] = Field(default_factory=list)


class ConversationSummary(BaseModel):
    title: str
    summary: str
    conversation_type: Literal["friendly", "professional", "mixed"]
    participants: list[str] = Field(default_factory=list)
    key_points: list[str] = Field(default_factory=list)
    action_items: list[str] = Field(default_factory=list)
    sentiment: Literal["positive", "neutral", "negative", "mixed"]
    duration_estimate_minutes: Optional[float] = None
