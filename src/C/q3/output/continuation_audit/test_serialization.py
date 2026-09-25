import json

import numpy as np

from build_continuation_audit import json_safe


def test_json_safe_converts_numpy_scalars():
    payload = {
        "bool": json_safe(np.bool_(True)),
        "int": json_safe(np.int64(4)),
        "float": json_safe(np.float64(1.25)),
    }
    assert json.loads(json.dumps(payload)) == {"bool": True, "int": 4, "float": 1.25}
