import importlib.util
import unittest
from pathlib import Path

import pandas as pd


SCRIPT_PATH = Path(__file__).with_name("9check_riemann_drop.py")
_module = None
if SCRIPT_PATH.exists():
    _spec = importlib.util.spec_from_file_location("check_riemann_drop", SCRIPT_PATH)
    _module = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_module)
select_trial_groups = getattr(_module, "select_trial_groups", None)
calculate_threshold = getattr(_module, "calculate_threshold", None)


class SelectTrialGroupsTests(unittest.TestCase):
    def require_selector(self):
        if select_trial_groups is None:
            self.fail("9check_riemann_drop.select_trial_groups is not implemented")
        return select_trial_groups

    def test_selects_lowest_and_nearest_five_trials(self):
        metrics = pd.DataFrame({
            "Trial": list(range(12)),
            "ClippedDrop": [0] * 12,
            "SQI": [0.02, 0.11, 0.20, 0.31, 0.42, 0.48,
                    0.50, 0.55, 0.62, 0.70, 0.80, 0.90],
            "RiemannDrop": [1] * 6 + [0] * 6,
        })

        groups, threshold = self.require_selector()(metrics, threshold=0.50)

        self.assertEqual(threshold, 0.50)
        self.assertEqual(groups["lowest_sqi"]["Trial"].tolist(), [0, 1, 2, 3, 4])
        self.assertEqual(groups["closest_below"]["Trial"].tolist(), [5, 4, 3, 2, 1])
        self.assertEqual(groups["closest_above"]["Trial"].tolist(), [6, 7, 8, 9, 10])

    def test_returns_all_below_threshold_trials_when_fewer_than_five(self):
        for below_count in (2, 3):
            with self.subTest(below_count=below_count):
                below_sqi = [0.1 * (index + 1) for index in range(below_count)]
                above_sqi = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
                metrics = pd.DataFrame({
                    "Trial": list(range(below_count + len(above_sqi))),
                    "ClippedDrop": [0] * (below_count + len(above_sqi)),
                    "SQI": below_sqi + above_sqi,
                    "RiemannDrop": [1] * below_count + [0] * len(above_sqi),
                })

                groups, threshold = self.require_selector()(metrics, threshold=0.5)

                self.assertEqual(threshold, 0.5)
                self.assertEqual(
                    len(groups["closest_below"]), below_count,
                    "the below-threshold group must contain every available trial, up to five",
                )
                self.assertEqual(
                    groups["closest_below"]["Trial"].tolist(),
                    list(range(below_count - 1, -1, -1)),
                )

    def test_excludes_clipped_trials_from_every_group(self):
        metrics = pd.DataFrame({
            "Trial": list(range(8)),
            "ClippedDrop": [1, 0, 0, 0, 0, 0, 0, 0],
            "SQI": [0.001, 0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8],
            "RiemannDrop": [0, 1, 1, 1, 0, 0, 0, 0],
        })

        groups, _ = self.require_selector()(metrics, threshold=0.5)

        selected_trials = {
            int(trial)
            for group in groups.values()
            for trial in group["Trial"].tolist()
        }
        self.assertNotIn(0, selected_trials)
        self.assertEqual(groups["lowest_sqi"]["Trial"].tolist()[:3], [1, 2, 3])

    def test_applies_group_b_conservative_factor_to_knee_threshold(self):
        if calculate_threshold is None:
            self.fail("9check_riemann_drop.calculate_threshold is not implemented")
        sqi = [0.01, 0.10, 0.11, 0.12, 0.13, 0.40, 0.41, 0.42, 0.43, 0.44]

        self.assertAlmostEqual(calculate_threshold(sqi, is_group_a=True), 0.10)
        self.assertAlmostEqual(calculate_threshold(sqi, is_group_a=False), 0.06)


if __name__ == "__main__":
    unittest.main()
