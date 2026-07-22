"""Self-check: load DAIC-WOZ interviews and print a summary."""

from __future__ import annotations

from pathlib import Path

from prefix_tuning_depression.data import load_interviews, print_interview_summary
from prefix_tuning_depression.splits import load_split, print_split_summary


def main() -> None:
    data_root = Path("data")

    split_map = load_split(data_root)
    print("=== Split summary ===")
    print_split_summary(split_map, data_root / "cleaning_report_Transcript.csv")
    print()

    print("=== Interview summary ===")
    interviews = load_interviews(data_root, split_map=split_map)
    print_interview_summary(interviews)

    sample = interviews[0]
    print()
    print(f"=== Sample interview ({sample.subject_id}) ===")
    print(f"PHQ score: {sample.phq_score}, QR pairs: {len(sample.qr_pairs)}")
    for i, pair in enumerate(sample.qr_pairs[:3], 1):
        print(f"  {i}. {pair[:160]}{'...' if len(pair) > 160 else ''}")


if __name__ == "__main__":
    main()
