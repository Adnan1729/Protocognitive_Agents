"""
End-to-end reproduction: runs every stage in order.

Run from the repository root, either way works:
    python -m scripts.reproduce_all
    python scripts/reproduce_all.py
"""

import sys
from pathlib import Path

# Make sibling script modules importable regardless of how this file is invoked.
# When run as `python scripts/reproduce_all.py`, the parent dir isn't on sys.path.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts import (  # noqa: E402
    plot_raw_spikes,
    transform_and_plot,
    compute_complexity,
    train_classifier,
)

STAGES = [
    ("Fig. 3  -- raw spike trains",           plot_raw_spikes.main),
    ("Fig. 4  -- multinodal graphs",          transform_and_plot.main),
    ("Table III, Fig. 5 -- complexity",       compute_complexity.main),
    ("Tables IV/V, Figs. 7-9 -- classifier",  train_classifier.main),
]


def main() -> None:
    for label, fn in STAGES:
        print(f"\n{'=' * 70}")
        print(f"Running: {label}")
        print("=" * 70)
        fn()


if __name__ == "__main__":
    main()