"""DAIC-WOZ transcript preprocessing — exact reproduction of Lau et al. (2023).

Processes raw official DAIC-WOZ transcripts (tab-separated:
``start_time | stop_time | speaker | value``) into question-response
pair strings ready for encoding.
"""

from __future__ import annotations

import re

import pandas as pd

# ---------------------------------------------------------------------------
# Hard-coded corrections from the paper repo
# ---------------------------------------------------------------------------

_interrupt: dict[int, list[float]] = {
    373: [398.0, 430.3],
    444: [285.6, 384.4],
}

_misaligned: dict[int, float] = {
    318: 34.319917,
    321: 3.8379167,
    341: 6.1892,
    362: 16.8582,
}

_START_PROMPTS = re.compile(
    r"are you okay with this|"
    r"hi i'm ellie thanks for coming in today|"
    r"think of me as a friend i don't judge|"
    r"and please feel free to tell me anything|"
    r"i'm not a therapist|"
    r"i'll ask a few questions|"
    r"i was created to talk",
)
_CLOSING_PROMPTS = re.compile(
    r"okay i think i have asked everything i need to|"
    r"okay i think i've asked everything i need to|"
    r"goodbye|"
    r"it was great chatting with you|"
    r"thanks for sharing your thoughts",
)


def _consecutive_groups(indices: list[int]) -> list[list[int]]:
    """Group consecutive integers.  Replaces ``more_itertools.consecutive_groups``."""
    if not indices:
        return []
    groups: list[list[int]] = []
    current = [indices[0]]
    for i in indices[1:]:
        if i == current[-1] + 1:
            current.append(i)
        else:
            groups.append(current)
            current = [i]
    groups.append(current)
    return groups


def _remove_unique_identifiers(df: pd.DataFrame) -> None:
    """Strip speaker names from Ellie lines — keep only parenthesised content."""
    mask = (df["speaker"] == "Ellie") & df["value"].str.contains("(", regex=False)
    for idx in df[mask].index:
        df.at[idx, "value"] = df.at[idx, "value"].split("(")[-1].rstrip(")")


def _remove_whitespace(df: pd.DataFrame) -> None:
    df["value"] = df["value"].str.strip()
    df["value"] = df["value"].str.replace(r" +", " ", regex=True)


def _remove_annotations(df: pd.DataFrame) -> None:
    sync_expr = r"<sync>|<sync\.|<synch>|\[syncing\]|\[sync\]|\[synch\]|\[synching\]"
    df.drop(
        df[df["value"].str.contains(sync_expr, na=False)].index,
        inplace=True,
    )
    df["value"] = df["value"].str.replace("<laughter>", "*laughter*", regex=False)
    df["value"] = df["value"].str.replace(r"\[laughter\]", "*laughter*", regex=True)
    df["value"] = df["value"].str.replace("<sigh>", "*sigh*", regex=False)
    df["value"] = df["value"].str.replace(r"<.*?>", "", regex=True)
    df["value"] = df["value"].str.replace(r"\[.*?\]", "", regex=True)
    df["value"] = df["value"].str.replace("scrubbed_entry", "", regex=False)
    df["value"] = df["value"].str.replace("xxxx", "", regex=False)
    df["value"] = df["value"].str.replace("xxx", "", regex=False)


def _remove_empty_rows(df: pd.DataFrame) -> None:
    df.drop(df[df["value"] == ""].index, inplace=True)
    df.drop(df[df["value"] == "laughter"].index, inplace=True)
    df.drop(df[df["value"] == "*laughter*"].index, inplace=True)
    df.reset_index(drop=True, inplace=True)


def _collapse_responses(df: pd.DataFrame) -> None:
    """Merge consecutive Participant lines with ``', '``.  Matches paper exactly."""
    participant_indices = df[df["speaker"] == "Participant"].index.tolist()
    groups = _consecutive_groups(participant_indices)
    multi_line_groups = [g for g in groups if len(g) > 1]

    for group in multi_line_groups:
        first_idx = group[0]
        last_idx = group[-1]
        start_time = df.at[first_idx, "start_time"]
        stop_time = df.at[last_idx, "stop_time"]
        value = ", ".join(df.at[i, "value"] for i in group)
        df.at[first_idx, "start_time"] = start_time
        df.at[first_idx, "stop_time"] = stop_time
        df.at[first_idx, "speaker"] = "Participant"
        df.at[first_idx, "value"] = value
        df.drop(group[1:], axis=0, inplace=True)


def _to_qr_pairs(df: pd.DataFrame) -> list[str]:
    """Pair each Participant response with the preceding Ellie question.

    Returns a list of concatenated QR-pair strings.
    """
    df = df.reset_index(drop=True)

    participant_idx = df[df["speaker"] == "Participant"].index.tolist()
    if not participant_idx:
        return []

    ellie_idx = df[df["speaker"] == "Ellie"].index.tolist()
    if not ellie_idx:
        return []

    # Drop Participant rows that appear before the first Ellie utterance.
    first_ellie = ellie_idx[0]
    participant_idx = [i for i in participant_idx if i > first_ellie]

    pairs: list[str] = []
    for p_idx in participant_idx:
        q_text = df.at[p_idx - 1, "value"]
        r_text = df.at[p_idx, "value"]
        pairs.append(f"{q_text} {r_text}")
    return pairs


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def preprocess_transcript(
    df: pd.DataFrame,
    subject_id: int | None = None,
) -> list[str]:
    """Apply the paper's full preprocessing pipeline and return QR pair strings.

    Args:
        df: Raw DAIC-WOZ transcript DataFrame with columns
            ``start_time``, ``stop_time``, ``speaker``, ``value``.
        subject_id: If provided, applies per-subject interruption and
            misalignment corrections from the paper repo.

    Returns:
        List of QR pair strings.
    """
    df = df.copy()

    # -- Per-subject corrections --------------------------------------------
    if subject_id is not None:
        if subject_id in _interrupt:
            onset_time, end_time = _interrupt[subject_id]
            mask_start = df["start_time"].astype(str).str.contains(str(onset_time))
            mask_end = df["stop_time"].astype(str).str.contains(str(end_time))
            if mask_start.any() and mask_end.any():
                onset_idx = df[mask_start].index.values[0]
                end_idx = df[mask_end].index.values[0]
                df.drop(df.index[onset_idx:end_idx], inplace=True)

        if subject_id in _misaligned:
            offset = _misaligned[subject_id]
            df["start_time"] = df["start_time"] + offset
            df["stop_time"] = df["stop_time"] + offset

    # -- Paper repo cleaning pipeline (order preserved) ---------------------
    df.dropna(inplace=True)
    _remove_annotations(df)
    _remove_whitespace(df)
    _remove_unique_identifiers(df)
    _remove_empty_rows(df)
    _collapse_responses(df)

    # Drop start / closing prompts.
    df.drop(
        df[df["value"].str.contains(_START_PROMPTS, regex=True, na=False)].index,
        inplace=True,
    )
    df.drop(
        df[df["value"].str.contains(_CLOSING_PROMPTS, regex=True, na=False)].index,
        inplace=True,
    )

    # Remove underscores used for acronyms, then lowercase.
    df["value"] = df["value"].str.replace("_", "", regex=False)
    df["value"] = df["value"].str.lower()

    return _to_qr_pairs(df)
