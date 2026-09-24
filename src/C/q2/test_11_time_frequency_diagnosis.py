"""Regression tests for the 11 time-frequency analysis pipeline."""

import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np


SCRIPT_PATH = Path(__file__).with_name("11_time_frequency_diagnosis.py")
SPEC = importlib.util.spec_from_file_location(
    "time_frequency_diagnosis",
    SCRIPT_PATH,
)
diagnosis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagnosis)


class BaselineOrderTests(unittest.TestCase):
    def test_post_onset_power_does_not_enter_prestimulus_baseline(self):
        time = np.arange(-200, 201, dtype=float) / 1000.0
        raw_power = np.ones((1, 3, len(time)), dtype=float)
        raw_power[..., time >= 0] = 4.0
        analytic_signal = np.sqrt(raw_power).astype(complex)
        trials = np.zeros_like(raw_power)

        with (
            patch.object(diagnosis, "BANDS", {"theta": (4.0, 8.0)}),
            patch.object(diagnosis, "butter", return_value="filter"),
            patch.object(
                diagnosis,
                "sosfiltfilt",
                side_effect=lambda _sos, values, axis: values,
            ),
            patch.object(
                diagnosis,
                "hilbert",
                return_value=analytic_signal,
            ),
        ):
            result = diagnosis.prepare_band_data(
                trials,
                time,
                fs=1000,
            )["theta"]

        baseline = (time >= -0.2) & (time < 0.0)
        onset = int(np.flatnonzero(time == 0.0)[0])

        np.testing.assert_allclose(result[..., baseline], 0.0, atol=1e-12)
        np.testing.assert_allclose(
            result[..., onset],
            10.0 * np.log10(4.0),
            atol=1e-12,
        )


if __name__ == "__main__":
    unittest.main()
