import unittest

import pandas as pd

from prefix_tuning_depression.preprocessing import preprocess_transcript


class TranscriptPreprocessingTests(unittest.TestCase):
    def test_pairs_only_adjacent_ellie_and_participant_turns(self) -> None:
        pairs = preprocess_transcript(
            self._transcript(
                [
                    ("Participant", "before ellie"),
                    ("Ellie", "First question?"),
                    ("Participant", "First answer"),
                    ("Ellie", "Goodbye"),
                    ("Participant", "orphan response"),
                    ("Ellie", "Second question?"),
                    ("Ellie", "Follow up question?"),
                    ("Participant", "Second answer"),
                    ("Ellie", "What did they say?"),
                    ("Participant", "They said goodbye"),
                ]
            )
        )

        self.assertEqual(
            pairs,
            [
                "first question? first answer",
                "follow up question? second answer",
                "what did they say? they said goodbye",
            ],
        )

    def test_normalizes_supported_nonverbal_annotations(self) -> None:
        pairs = preprocess_transcript(
            self._transcript(
                [
                    ("Ellie", "How are you?"),
                    (
                        "Participant",
                        "<sigh> [laughter] {hsighi} ((laughter)) l_a <cut>",
                    ),
                ]
            )
        )

        self.assertEqual(
            pairs,
            ["how are you? *sigh* *laughter* *sigh* *laughter* la"],
        )

    def _transcript(self, turns: list[tuple[str, str]]) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "start_time": range(len(turns)),
                "stop_time": range(1, len(turns) + 1),
                "speaker": [speaker for speaker, _ in turns],
                "value": [value for _, value in turns],
            }
        )
