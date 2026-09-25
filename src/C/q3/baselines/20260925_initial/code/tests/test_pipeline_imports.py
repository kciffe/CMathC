from pathlib import Path
import runpy


def test_raw_preprocessing_script_imports_all_runtime_constants():
    script = Path(__file__).resolve().parents[1] / "02b_preprocess_raw.py"

    namespace = runpy.run_path(str(script), run_name="q3_raw_preprocess_import_test")

    assert "RAW_SAMPLE_RATE_HZ" in namespace
    assert callable(namespace["main"])
