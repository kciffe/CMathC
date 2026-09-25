import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from threshold_search import select_threshold_candidate


def test_threshold_selection_uses_inner_score_then_stable_candidate_order():
    scored = [
        {"inner_mean_ba": 0.55, "candidate_order": 0, "threshold": 0.5, "candidate": "base"},
        {"inner_mean_ba": 0.60, "candidate_order": 1, "threshold": 0.4, "candidate": "better"},
        {"inner_mean_ba": 0.60, "candidate_order": 2, "threshold": 0.5, "candidate": "later"},
    ]

    assert select_threshold_candidate(scored)["candidate"] == "better"
