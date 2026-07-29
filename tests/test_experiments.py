import unittest

from prefix_tuning_depression.data import Interview
from prefix_tuning_depression.experiments import best_run, stratified_train_subset


class ExperimentUtilityTests(unittest.TestCase):
    def test_stratified_subset_preserves_groups_and_seed(self) -> None:
        interviews = [
            Interview(
                subject_id=index,
                qr_pairs=["q r"],
                phq_score=0,
                phq_binary=index % 2,
                split="train",
            )
            for index in range(20)
        ]

        subset = stratified_train_subset(interviews, percentage=50, random_state=7)

        self.assertEqual(len(subset), 10)
        self.assertEqual(sum(interview.phq_binary for interview in subset), 5)
        self.assertEqual(
            [interview.subject_id for interview in subset],
            [
                interview.subject_id
                for interview in stratified_train_subset(
                    interviews, percentage=50, random_state=7
                )
            ],
        )

    def test_selects_the_lowest_development_loss(self) -> None:
        self.assertEqual(
            best_run([{"loss": 2.0}, {"loss": 1.0}]),
            {"loss": 1.0},
        )
