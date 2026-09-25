import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from model_search import grouped_leave_one_group_out, scale_train_test


def test_grouped_folds_hold_out_whole_records():
    groups = np.array(["r1"] * 4 + ["r2"] * 4 + ["r3"] * 4)
    folds = grouped_leave_one_group_out(groups)

    assert len(folds) == 3
    for train_index, test_index, held_out_group in folds:
        assert set(groups[train_index]).isdisjoint(set(groups[test_index]))
        assert set(groups[test_index]) == {held_out_group}


def test_scaler_uses_training_rows_only():
    train = np.array([[0.0, 2.0], [2.0, 4.0]])
    test = np.array([[100.0, -100.0]])
    scaled_train, scaled_test, center, scale = scale_train_test(train, test, "standard")

    np.testing.assert_allclose(center, [1.0, 3.0])
    np.testing.assert_allclose(scale, [1.0, 1.0])
    np.testing.assert_allclose(scaled_train.mean(axis=0), [0.0, 0.0])
    np.testing.assert_allclose(scaled_test, [[99.0, -103.0]])


def test_robust_scaler_uses_training_median_and_iqr():
    train = np.array([[0.0], [2.0], [4.0], [100.0]])
    test = np.array([[1000.0]])
    _, scaled_test, center, scale = scale_train_test(train, test, "robust")

    np.testing.assert_allclose(center, [3.0])
    np.testing.assert_allclose(scale, [26.5])
    np.testing.assert_allclose(scaled_test, [[997.0 / 26.5]])
