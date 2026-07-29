"""Self-check: load DAIC-WOZ interviews and print a summary."""

from __future__ import annotations

from prefix_tuning_depression.data import load_interviews, print_interview_summary
from prefix_tuning_depression.splits import (
    load_official_avec2017_contract,
    require_complete_official_transcript_coverage,
)


def main() -> None:
    manifest_dir = "data/transcript"
    contract = load_official_avec2017_contract(manifest_dir)
    require_complete_official_transcript_coverage(contract, manifest_dir)

    print("=== Split summary ===")
    for split in ("train", "dev", "test"):
        print(f"{split}: {sum(value == split for value in contract.split_map.values())}")
    print()

    print("=== Interview summary ===")
    interviews = load_interviews(manifest_dir, contract=contract)
    print_interview_summary(interviews)

    sample = interviews[0]
    print()
    print(f"=== Sample interview ({sample.subject_id}) ===")
    print(f"PHQ score: {sample.phq_score}, QR pairs: {len(sample.qr_pairs)}")
    for i, pair in enumerate(sample.qr_pairs[:3], 1):
        print(f"  {i}. {pair[:160]}{'...' if len(pair) > 160 else ''}")


if __name__ == "__main__":
    main()
