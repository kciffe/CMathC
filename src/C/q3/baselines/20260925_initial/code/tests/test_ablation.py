import numpy as np
import pandas as pd

from ablation import make_proxy_features


def _frame(n=8, shift=0.0):
    base = np.arange(n, dtype=float) + shift
    frame = pd.DataFrame({
        "erp_mean_F3": base,
        "erp_mean_Fz": base + 1,
        "erp_mean_F4": base + 2,
        "log_gamma_power_F3": base + 3,
        "log_gamma_power_Fz": base + 4,
        "log_gamma_power_F4": base + 5,
        "log_theta_power_F3": base + 6,
        "log_theta_power_Fz": base + 7,
        "log_theta_power_F4": base + 8,
        "log_beta_power_F3": base + 9,
        "log_beta_power_Fz": base + 10,
        "log_beta_power_F4": base + 11,
        "AI_alpha": base + 12,
    })
    return frame


def test_proxy_scaling_uses_training_statistics_only():
    train = _frame()
    test = _frame(shift=1000.0)

    train_proxy, test_proxy = make_proxy_features(train, test)

    np.testing.assert_allclose(train_proxy.mean(axis=0).to_numpy(), 0.0, atol=1e-12)
    assert (test_proxy.mean(axis=0) > 100.0).all()


def test_reconstructing_a_feature_excludes_it_from_its_anchor_proxy():
    train = _frame()
    test_a = _frame(n=2)
    test_b = test_a.copy()
    test_b["erp_mean_F3"] = 1e9

    _, proxy_a = make_proxy_features(train, test_a, excluded_feature="erp_mean_F3")
    _, proxy_b = make_proxy_features(train, test_b, excluded_feature="erp_mean_F3")

    np.testing.assert_allclose(proxy_a["V"], proxy_b["V"])
