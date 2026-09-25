import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from temporal_search import baseline_correct_signal, erp_bin_features, fit_csp_filters, transform_csp


def test_baseline_correction_uses_only_pre_event_samples():
    time = np.arange(-0.1, 0.5, 1 / 100)
    signal = np.vstack([np.where(time < 0, 10.0, 12.0), np.where(time < 0, -4.0, -1.0)])

    corrected = baseline_correct_signal(signal, time)

    np.testing.assert_allclose(corrected[:, time < 0].mean(axis=1), [0.0, 0.0])
    np.testing.assert_allclose(corrected[:, time >= 0].mean(axis=1), [2.0, 3.0])


def test_erp_bins_return_temporal_means_in_channel_major_order():
    time = np.arange(-0.1, 0.5, 1 / 100)
    signal = np.vstack([np.where(time < -1e-9, 5.0, 7.0), np.where(time < -1e-9, -2.0, 1.0)])
    corrected = baseline_correct_signal(signal, time)

    values, names = erp_bin_features(corrected, time, duration_s=0.5, bin_width_s=0.1)

    assert len(values) == 10
    assert len(names) == 10
    np.testing.assert_allclose(values, [2.0] * 5 + [3.0] * 5)


def test_csp_filters_are_fitted_from_training_trials_and_transform_finitely():
    rng = np.random.default_rng(14)
    n = 40
    n_time = 128
    labels = np.tile(np.array([-1, 1]), n // 2)
    signals = rng.normal(scale=0.25, size=(n, 3, n_time))
    signals[labels == -1, 0] += rng.normal(scale=2.0, size=(int((labels == -1).sum()), n_time))
    signals[labels == 1, 2] += rng.normal(scale=2.0, size=(int((labels == 1).sum()), n_time))

    filters = fit_csp_filters(signals, labels, edge_components=1, shrinkage=0.01)
    features = transform_csp(signals, filters)

    assert filters.shape == (2, 3)
    assert features.shape == (n, 2)
    assert np.isfinite(features).all()
