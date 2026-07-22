"""Transcript preprocessing and QR pair grouping.

The cleaned transcripts available here do not contain a speaker column, so we use
a heuristic to reconstruct Ellie / Participant turns and group them into
question-response pairs as described in Lau et al. (2023).
"""

from __future__ import annotations

import re

import pandas as pd

# Regex for start-of-interview and end-of-interview prompts to remove.
_START_PROMPTS_RE = re.compile(
    r"are you okay with this|"
    r"hi i'm ellie thanks for coming in today|"
    r"think of me as a friend i don't judge|"
    r"and please feel free to tell me anything|"
    r"i'm not a therapist|"
    r"i'll ask a few questions|"
    r"i was created to talk"
)
_CLOSING_PROMPTS_RE = re.compile(
    r"okay i think i have asked everything i need to|"
    r"okay i think i've asked everything i need to|"
    r"goodbye|"
    r"it was great chatting with you|"
    r"thanks for sharing your thoughts"
)

# Heuristic cues used to label utterances without an explicit speaker column.
_QUESTION_CUES = (
    "what", "where", "how", "when", "why", "who", "which",
    "do you", "are you", "did you", "have you", "has you", "can you",
    "would you", "could you", "will you", "were you", "was you",
)
_PARTICIPANT_CUES = (
    "i ", "i'm", "i've", "i'd", "i'll",
    "my ", "me ", "myself", "we ", "we're", "our ",
)


def _is_question_like(text: str) -> bool:
    """Return True if the utterance looks like an interviewer question."""
    lowered = text.lower()
    if lowered.endswith("?"):
        return True
    return any(cue in lowered for cue in _QUESTION_CUES)


def _is_participant_like(text: str) -> bool:
    """Return True if the utterance looks like a participant response."""
    lowered = text.lower()
    return any(cue in lowered for cue in _PARTICIPANT_CUES)


def _label_speakers(utterances: list[str]) -> list[str]:
    """Assign 'Ellie' or 'Participant' labels using alternating + keyword cues.

    The first non-empty utterance is treated as Ellie. Consecutive utterances
    that share the same predicted speaker are allowed; downstream collapse logic
    merges consecutive Participant utterances.
    """
    labels: list[str] = []
    last_speaker = "Participant"  # will flip to Ellie on first utterance

    for text in utterances:
        if not text.strip():
            labels.append(last_speaker)
            continue

        if _is_question_like(text):
            speaker = "Ellie"
        elif _is_participant_like(text):
            speaker = "Participant"
        else:
            speaker = "Ellie" if last_speaker == "Participant" else "Participant"

        labels.append(speaker)
        last_speaker = speaker

    return labels


def clean_text(text: str) -> str:
    """Apply the paper's transcript cleaning rules.

    - Standardize laughter / sigh annotations.
    - Remove sync markers, scrubbed entries, unintelligible markers.
    - Remove angle-bracket and square-bracket annotations.
    - Lowercase and strip whitespace.
    - Remove underscores used for acronyms.
    """
    text = text.strip()

    # Standardize common non-verbal annotations first.
    text = re.sub(r"\{?hlaughteri\}?|\[laughter\]|<laughter>", "*laughter*", text, flags=re.IGNORECASE)
    text = re.sub(r"\{?hsighi\}?|<sigh>", "*sigh*", text, flags=re.IGNORECASE)

    # Remove specific markers.
    text = re.sub(r"\[syncing\]|\[sync\]|\[synch\]|\[synching\]", "", text, flags=re.IGNORECASE)
    text = re.sub(r"<sync>|<sync\.|<synch>|<synching>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bscrubbed_entry\b|\bxxxx\b|\bxxx\b", "", text, flags=re.IGNORECASE)

    # Remove remaining angle-bracket and square-bracket annotations.
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"\[.*?\]", "", text)

    # Remove underscores used for acronyms.
    text = text.replace("_", "")

    # Collapse whitespace.
    text = re.sub(r"\s+", " ", text).strip()

    return text.lower()


def remove_prompts(utterances: list[str]) -> list[str]:
    """Drop known start-of-interview and end-of-interview prompts."""
    return [
        u for u in utterances
        if not _START_PROMPTS_RE.search(u.lower())
        and not _CLOSING_PROMPTS_RE.search(u.lower())
    ]


def collapse_participant_turns(
    utterances: list[str], speakers: list[str]
) -> tuple[list[str], list[str]]:
    """Merge consecutive Participant utterances into single turns.

    Returns the collapsed utterance list and corresponding speaker list.
    """
    collapsed_utterances: list[str] = []
    collapsed_speakers: list[str] = []

    for text, speaker in zip(utterances, speakers):
        if speaker == "Participant" and collapsed_speakers and collapsed_speakers[-1] == "Participant":
            collapsed_utterances[-1] = collapsed_utterances[-1] + " " + text
        else:
            collapsed_utterances.append(text)
            collapsed_speakers.append(speaker)

    return collapsed_utterances, collapsed_speakers


def to_qr_pairs(utterances: list[str], speakers: list[str]) -> list[str]:
    """Group consecutive Ellie -> Participant utterances into QR pairs.

    Each pair is returned as 'question response'.
    """
    qr_pairs: list[str] = []
    last_ellie: str | None = None

    for text, speaker in zip(utterances, speakers):
        if speaker == "Ellie":
            last_ellie = text
        elif speaker == "Participant" and last_ellie is not None:
            qr_pairs.append(f"{last_ellie} {text}")
            last_ellie = None

    return qr_pairs


def preprocess_transcript(df: pd.DataFrame) -> list[str]:
    """Convert a cleaned transcript DataFrame into a list of QR pair strings."""
    utterances = [clean_text(str(t)) for t in df["Text"].tolist()]
    utterances = [u for u in utterances if u.strip()]
    utterances = remove_prompts(utterances)

    speakers = _label_speakers(utterances)
    utterances, speakers = collapse_participant_turns(utterances, speakers)
    qr_pairs = to_qr_pairs(utterances, speakers)

    return qr_pairs
