import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from joint_search import select_best_by_inner


def test_joint_selection_uses_inner_score_and_stable_predeclared_tie_break():
    scored = [(0.55, 0, "older_candidate"), (0.60, 1, "temporal_candidate"), (0.60, 2, "later_tie")]

    assert select_best_by_inner(scored) == (0.60, 1, "temporal_candidate")
