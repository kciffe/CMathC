"""Tests for B-group SQI sensitivity analysis helpers."""

import importlib.util
from pathlib import Path
import unittest

import numpy as np


MODULE_PATH = Path(__file__).with_name("10check_b_alpha_sensitivity.py")
SPEC = importlib.util.spec_from_file_location("b_alpha_sensitivity", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class AlphaThresholdTests(unittest.TestCase):
    def test_threshold_uses_alpha_and_floor(self):
        self.assertAlmostEqual(MODULE.threshold_for_alpha(0.4, 0.6), 0.24)
        self.assertAlmostEqual(MODULE.threshold_for_alpha(0.4, 1.0), 0.4)
        self.assertAlmostEqual(MODULE.threshold_for_alpha(0.06, 0.6), 0.05)

    def test_threshold_changes_never_retain_clipped_trials(self):
        sqi = np.array([0.9, 0.3, 0.2])
        clipped = np.array([True, False, False])

        for threshold in (0.15, 0.25, 0.35):
            keep = MODULE.retained_mask_for_threshold(sqi, clipped, threshold)
            self.assertFalse(keep[0])
            self.assertTrue(np.all(keep <= ~clipped))


class ErpMetricTests(unittest.TestCase):
    def test_baseline_corrects_each_trial_and_averages_by_cue(self):
        time = np.array([-0.2, -0.1, 0.0, 0.25, 0.5, 2.45, 2.7])
        # Two channels, three trials. t=0 is deliberately excluded from baseline.
        trials = np.array([
            [[2, 4, 100, 12, 22, 32, 42], [0, 2, 10, 8, 8, 8, 8]],
            [[4, 6, 200, 24, 44, 64, 84], [2, 4, 20, 16, 16, 16, 16]],
            [[8, 10, 300, 38, 58, 68, 78], [4, 6, 30, 20, 20, 20, 20]],
        ], dtype=float)
        relative_time = np.tile(time, (3, 1))
        cue_type = np.array([-1, -1, 1])
        keep = np.array([True, True, True])

        corrected = MODULE.baseline_correct_trials(trials, relative_time)
        erps = MODULE.average_by_condition(corrected, cue_type, keep)

        # Baselines are [3, 5, 9] per trial/channel, not affected by the t=0 samples.
        np.testing.assert_allclose(corrected[0, 0], [-1, 1, 97, 9, 19, 29, 39])
        np.testing.assert_allclose(erps[-1], np.mean(corrected[:2], axis=0))
        np.testing.assert_allclose(erps[1], corrected[2])

    def test_candidate_peaks_return_cue_and_target_relative_latency(self):
        time = np.array([0.25, 0.3, 0.5, 2.45, 2.55, 2.7])
        fz_erp = np.array([2.0, 5.0, 4.0, 3.0, 8.0, 7.0])

        metrics = MODULE.extract_candidate_peaks(fz_erp, time)

        self.assertEqual(metrics["cue_peak_amplitude"], 5.0)
        self.assertEqual(metrics["cue_latency_relative_to_cue_s"], 0.3)
        self.assertEqual(metrics["target_peak_amplitude"], 8.0)
        self.assertEqual(metrics["target_latency_relative_to_cue_s"], 2.55)
        self.assertAlmostEqual(metrics["target_latency_relative_to_target_s"], 0.35)

    def test_alpha_point_six_must_match_formal_final_drop(self):
        sqi = np.array([0.1, 0.3, 0.4, np.nan])
        clipped = np.array([False, False, False, True])
        formal_final_drop = np.array([True, False, False, True])

        keep = MODULE.validate_formal_alpha06(
            sqi, clipped, formal_final_drop, threshold=0.25
        )

        np.testing.assert_array_equal(keep, [False, True, True, False])
        with self.assertRaisesRegex(ValueError, "alpha=0.6"):
            MODULE.validate_formal_alpha06(
                sqi, clipped, np.array([True, True, False, True]), threshold=0.25
            )


if __name__ == "__main__":
    unittest.main()
