import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from expanded_search import fit_selector_train_only


def test_feature_selection_uses_training_labels_and_preserves_eval_rows():
    train = np.array([[0, 0, 1, 4], [1, 1, 2, 3], [8, 0, 1, 0], [9, 1, 2, 1]], dtype=float)
    labels = np.array([-1, -1, 1, 1])
    evaluate = np.array([[0.5, 0, 1.5, 3.5], [8.5, 1, 1.5, 0.5]], dtype=float)

    x_train, x_eval, selected = fit_selector_train_only(train, labels, evaluate, k=1)

    assert selected.tolist() == [0]
    assert x_train.shape == (4, 1)
    assert x_eval.shape == (2, 1)
    np.testing.assert_array_equal(x_eval[:, 0], evaluate[:, 0])
