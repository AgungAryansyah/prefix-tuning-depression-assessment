"""Audit official DAIC-WOZ readiness before expensive training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from prefix_tuning_depression.audit import build_reproduction_audit
from prefix_tuning_depression.splits import load_official_avec2017_contract


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit official reproduction inputs")
    parser.add_argument("--manifest-dir", type=Path, required=True)
    args = parser.parse_args()

    contract = load_official_avec2017_contract(args.manifest_dir)
    audit = build_reproduction_audit(contract, args.manifest_dir)
    print(json.dumps(audit, indent=2))
    return 0 if audit["status"] == "ready" else 2


if __name__ == "__main__":
    raise SystemExit(main())
