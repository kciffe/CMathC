import unittest
import importlib.util
from pathlib import Path

import numpy as np

MODULE_PATH = Path(__file__).with_name("6EEG_slice_drop_clipped_trials.py")
SPEC = importlib.util.spec_from_file_location("eeg_slice_drop_clipped_trials", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
slice_trials_with_drop = MODULE.slice_trials_with_drop


class SliceTrialsWithDropTests(unittest.TestCase):
    def test_keeps_all_trials_and_marks_eeg_clipping(self):
        fs = 2
        data = np.zeros((10, 30), dtype=float)
        data[9] = np.arange(30) / fs
        data[7, 4] = -1
        data[7, 20] = 1

        data[1, 5] = 999
        data[3, 21] = 999

        result = slice_trials_with_drop(data, fs)

        self.assertEqual(result["trial_data"].shape, (2, 10, 8))
        np.testing.assert_array_equal(result["drop"], np.array([1, 0]))
        self.assertNotIn("event_index", result)
        np.testing.assert_allclose(
            result["relative_time"],
            np.array([
                [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5],
                [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5],
            ]),
        )


if __name__ == "__main__":
    unittest.main()
