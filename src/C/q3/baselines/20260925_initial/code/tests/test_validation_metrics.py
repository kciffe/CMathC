from pathlib import Path
import runpy

import numpy as np


def test_auc_and_macro_f1_metrics_match_known_binary_example():
    script = Path(__file__).resolve().parents[1] / "06_validate.py"
    namespace = runpy.run_path(str(script), run_name="q3_validate_metric_test")

    auc = namespace["area_under_curve"](
        np.asarray([0, 0, 1, 1]), np.asarray([0.1, 0.4, 0.35, 0.8])
    )
    macro_f1 = namespace["macro_f1"](
        np.asarray([-1, -1, 1, 1]), np.asarray([-1, 1, 1, 1])
    )

    assert auc == 0.75
    assert np.isclose(macro_f1, (2 / 3 + 0.8) / 2)
