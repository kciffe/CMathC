from __future__ import annotations

import importlib
import importlib.util

import numpy as np


def _validation_module():
    spec = importlib.util.find_spec("dynamic_validation")
    assert spec is not None, "held-out dynamic validation utilities have not been implemented"
    return importlib.import_module("dynamic_validation")


def test_fixed_epoch_interpolates_to_shared_cue_locked_grid():
    validation = _validation_module()
    fs = 256.0
    time = np.arange(0.0, 5.0, 1.0 / fs)
    signal = np.stack([np.sin(2 * np.pi * 3 * time + phase) for phase in (0.0, 0.2, 0.4)])
    epoch, relative_time = validation.extract_fixed_epoch(
        signal,
        time,
        cue_time_s=1.8,
        next_cue_time_s=4.9,
        start_s=-0.2,
        stop_s=3.0,
        target_fs_hz=250.0,
    )

    assert epoch.shape == (3, 801)
    assert relative_time.shape == (801,)
    assert np.isclose(relative_time[0], -0.2)
    assert np.isclose(relative_time[-1], 3.0)
    assert np.isfinite(epoch).all()


def test_train_only_sensor_mapping_recovers_added_memory_signal():
    validation = _validation_module()
    rng = np.random.default_rng(30)
    groups, n_time = 6, 500
    q2 = rng.normal(size=(groups, 3, n_time))
    memory = rng.normal(size=(groups, n_time))
    control = rng.normal(size=(groups, n_time))
    loadings = np.array([0.8, -0.4, 0.6])
    observed = 0.3 * q2 + loadings[None, :, None] * memory[:, None, :]
    observed += rng.normal(scale=0.03, size=observed.shape)

    fit_visual = validation.fit_sensor_mapping(
        observed, q2, memory, control, include_memory=False, include_control=False
    )
    fit_memory = validation.fit_sensor_mapping(
        observed, q2, memory, control, include_memory=True, include_control=False
    )
    pred_visual = validation.predict_sensor_mapping(fit_visual, q2, memory, control)
    pred_memory = validation.predict_sensor_mapping(fit_memory, q2, memory, control)

    err_visual = np.mean((observed - pred_visual) ** 2)
    err_memory = np.mean((observed - pred_memory) ** 2)
    assert err_memory < err_visual * 0.25


def test_model_mapping_rejects_unaligned_shapes():
    validation = _validation_module()
    observed = np.zeros((2, 3, 20))
    q2 = np.zeros((2, 3, 20))
    memory = np.zeros((2, 19))
    control = np.zeros((2, 20))

    try:
        validation.fit_sensor_mapping(observed, q2, memory, control)
    except ValueError as error:
        assert "align" in str(error).lower()
    else:
        raise AssertionError("misaligned state and EEG arrays must fail clearly")


def test_training_template_uses_only_training_groups_for_matching_cue_side():
    validation = _validation_module()
    training = np.stack([
        np.full((3, 4), 2.0),
        np.full((3, 4), 6.0),
        np.full((3, 4), -3.0),
    ])
    train_sides = np.array([-1, -1, 1])
    test_sides = np.array([-1, 1])

    prediction = validation.predict_training_side_template(
        training, train_sides, test_sides
    )

    assert prediction.shape == (2, 3, 4)
    np.testing.assert_allclose(prediction[0], 4.0)
    np.testing.assert_allclose(prediction[1], -3.0)


def test_continuous_filter_operator_preserves_shape_and_none_is_identity():
    validation = _validation_module()
    fs = 256.0
    time = np.arange(2048) / fs
    signal = np.stack([
        np.sin(2 * np.pi * 5 * time),
        np.sin(2 * np.pi * 10 * time),
        np.sin(2 * np.pi * 20 * time),
    ])

    unchanged = validation.filter_continuous_eeg(
        signal, fs, validation.macro.PreprocessingConfig(mode="none")
    )
    filtered = validation.filter_continuous_eeg(
        signal, fs, validation.macro.PreprocessingConfig(mode="zero_phase")
    )

    np.testing.assert_allclose(unchanged, signal, atol=0.0)
    assert filtered.shape == signal.shape
    assert np.isfinite(filtered).all()
    assert not np.allclose(filtered, signal)
