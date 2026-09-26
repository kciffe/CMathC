"""Audit Q3 trial truth, target timing, response outcomes, and record metadata.

Run from the repository root after scripts 01_audit_events.py and
02_extract_trials.py have produced ``src/C/q3/output/trial_table.csv``::

    python src/C/q3/08_behavior_event_audit.py

If external input tables are missing, this script creates keyed CSV templates
under ``src/C/q3/input/`` and still audits channel-derived correctness and
cue-interval omissions. Target onsets, deadlines, and record identities remain
unknown unless independently supplied.

All event times must use the same absolute recording clock as the MAT
``TimeStamp`` channel. See ``src/C/q3/input/README_外部信息表说明.md`` for the
input schema and outcome rules.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


Q3_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(Q3_DIR))

from common import load_raw_record  # noqa: E402
from config import INPUT_DIR, OUTPUT_DIR  # noqa: E402


KEYS = ["record", "original_trial_index"]
TRUTH_FILE = "trial_truth.csv"
EVENT_FILE = "event_log.csv"
MAPPING_FILE = "record_mapping.csv"
README_FILE = "README_外部信息表说明.md"


def _blank(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value)) or str(value).strip() == ""
    except (TypeError, ValueError):
        return False


def _parse_side(value: Any, *, field: str) -> float:
    """Return -1/+1 for an explicit left/right value, or NaN for blank."""
    if _blank(value):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        number = float(value)
        if number in (-1.0, 1.0):
            return number
    text = str(value).strip().lower()
    if text in {"left", "l", "-1", "左"}:
        return -1.0
    if text in {"right", "r", "+1", "1", "右"}:
        return 1.0
    raise ValueError(f"{field} must be left/right or -1/+1; got {value!r}")


def _parse_optional_bool(value: Any, *, field: str) -> float:
    """Parse an optional boolean into 1, 0, or NaN."""
    if _blank(value):
        return np.nan
    if isinstance(value, (bool, np.bool_)):
        return float(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        number = float(value)
        if number in (0.0, 1.0):
            return number
    text = str(value).strip().lower()
    if text in {"true", "yes", "y", "1", "是", "有"}:
        return 1.0
    if text in {"false", "no", "n", "0", "否", "无"}:
        return 0.0
    raise ValueError(f"{field} must be true/false, yes/no, or 1/0; got {value!r}")


def _normalize_keys(frame: pd.DataFrame, source_name: str) -> pd.DataFrame:
    missing = sorted(set(KEYS).difference(frame.columns))
    if missing:
        raise ValueError(f"{source_name} is missing join columns: {missing}")
    out = frame.copy()
    out["record"] = out["record"].astype("string").str.strip()
    trial_num = pd.to_numeric(out["original_trial_index"], errors="coerce")
    invalid = trial_num.isna() | (trial_num % 1 != 0)
    if invalid.any():
        raise ValueError(
            f"{source_name} has {int(invalid.sum())} invalid original_trial_index values"
        )
    out["original_trial_index"] = trial_num.astype(int)
    if out["record"].isna().any() or out["record"].eq("").any():
        raise ValueError(f"{source_name} has a blank record key")
    return out


def _assert_unique_keys(frame: pd.DataFrame, source_name: str) -> None:
    duplicated = frame.duplicated(KEYS, keep=False)
    if duplicated.any():
        sample = frame.loc[duplicated, KEYS].head(5).to_dict(orient="records")
        raise ValueError(f"{source_name} has duplicate record/trial keys: {sample}")


def _write_csv_if_missing(path: Path, frame: pd.DataFrame) -> bool:
    if path.exists():
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    return True


def _write_input_readme(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "# Q3 外部行为与事件信息表说明\n\n"
        "所有 CSV 均使用 UTF-8 编码，按 `record + original_trial_index` 与\n"
        "`src/C/q3/output/trial_table.csv` 对齐。试次编号从 0 开始。不要按行号\n"
        "猜测对应关系。保留空值表示未知，不要填 0 代替未知。\n\n"
        "## trial_truth.csv\n\n"
        "该表为可选的独立核验表，列为 `record`、`original_trial_index`、\n"
        "`correct_response_side`、`truth_source`、`authoritative_omission`。\n"
        "题目已定义通道8 VisCue 为目标侧（-1左、+1右），主分析直接用通道8；\n"
        "此表只用于独立核对，不是生成正确/错误标签的前置条件。\n\n"
        "## event_log.csv\n\n"
        "真实事件日志是可选输入。没有真实 target marker 时，脚本沿用 trial_table.csv\n"
        "已有 cue+2.2 s 排程锚点，并明确标记为假设值，不称为真实 target onset。\n"
        "若有真实事件日志，推荐宽表：每试次一行，列为\n"
        "`record, original_trial_index, target_onset_time_s, "
        "response_deadline_time_s, timing_source`。时间是 MAT `TimeStamp` 的同一\n"
        "绝对秒时钟；截止时间填截止时刻的绝对时间，不填持续时长。截止时间可以\n"
        "留空。也接受长表：`record, original_trial_index, event_type, timestamp_s`，\n"
        "其中 event_type 使用 `target_onset` 或 `response_deadline`。\n\n"
        "## record_mapping.csv（可选）\n\n"
        "每份 MAT 一行：`record, participant_id, task_id, session_id, mapping_source`。\n"
        "按实验登记表填写，不能根据文件名推断被试或真实任务编号。编码信息会另从\n"
        "MAT 的 DataLabel、VisCue 事件取值和当前事件解析代码中审计。\n\n"
        "## 判定口径\n\n"
        "脚本将通道8目标侧与通道9首个动作侧比较，生成正确/错误标签。\n"
        "某 VisCue 到下一 VisCue 区间没有通道9动作边沿，则按本项目口径记为区间漏答；\n"
        "迟答不单独分类。目标 onset 缺失时不计算真实反应时。\n",
        encoding="utf-8",
    )


def create_missing_input_templates(
    input_dir: Path, trial_table: pd.DataFrame
) -> list[Path]:
    """Create keyed templates without overwriting any user-provided input."""
    input_dir.mkdir(parents=True, exist_ok=True)
    keys = trial_table[KEYS].drop_duplicates().sort_values(KEYS).reset_index(drop=True)
    created: list[Path] = []
    truth_path = input_dir / TRUTH_FILE
    if _write_csv_if_missing(
        truth_path,
        keys.assign(
            correct_response_side="",
            truth_source="",
            authoritative_omission="",
        ),
    ):
        created.append(truth_path)

    event_path = input_dir / EVENT_FILE
    if _write_csv_if_missing(
        event_path,
        keys.assign(
            target_onset_time_s="",
            response_deadline_time_s="",
            timing_source="",
        ),
    ):
        created.append(event_path)

    mapping_path = input_dir / MAPPING_FILE
    records = sorted(trial_table["record"].dropna().astype(str).unique())
    if _write_csv_if_missing(
        mapping_path,
        pd.DataFrame(
            {
                "record": records,
                "participant_id": [""] * len(records),
                "task_id": [""] * len(records),
                "session_id": [""] * len(records),
                "mapping_source": [""] * len(records),
            }
        ),
    ):
        created.append(mapping_path)

    readme_path = input_dir / README_FILE
    if not readme_path.exists():
        _write_input_readme(readme_path)
        created.append(readme_path)
    return created


def _read_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)


def load_truth_table(path: Path) -> pd.DataFrame:
    truth = _normalize_keys(_read_csv(path, "trial truth table"), TRUTH_FILE)
    _assert_unique_keys(truth, TRUTH_FILE)
    if "correct_response_side" not in truth:
        raise ValueError(f"{TRUTH_FILE} needs a correct_response_side column")
    truth["correct_response_side"] = [
        _parse_side(value, field="correct_response_side")
        for value in truth["correct_response_side"]
    ]
    if "authoritative_omission" in truth:
        truth["authoritative_omission"] = [
            _parse_optional_bool(value, field="authoritative_omission")
            for value in truth["authoritative_omission"]
        ]
    else:
        truth["authoritative_omission"] = np.nan
    if "truth_source" not in truth:
        truth["truth_source"] = ""
    return truth[
        KEYS + ["correct_response_side", "truth_source", "authoritative_omission"]
    ]


def _event_name(value: Any) -> str:
    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "target": "target_onset",
        "targetonset": "target_onset",
        "target_onset_s": "target_onset",
        "response_deadline_s": "response_deadline",
        "deadline": "response_deadline",
        "deadline_time": "response_deadline",
    }
    return aliases.get(text, text)


def load_event_table(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    events = _normalize_keys(_read_csv(path, "target event log"), EVENT_FILE)
    diagnostics: dict[str, Any] = {"format": None, "ignored_event_types": []}
    if {"event_type", "timestamp_s"}.issubset(events.columns):
        events["event_type_normalized"] = events["event_type"].map(_event_name)
        recognized = {"target_onset", "response_deadline"}
        diagnostics["format"] = "long"
        diagnostics["ignored_event_types"] = sorted(
            events.loc[~events["event_type_normalized"].isin(recognized), "event_type"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )
        relevant = events.loc[events["event_type_normalized"].isin(recognized)].copy()
        relevant["timestamp_s"] = pd.to_numeric(relevant["timestamp_s"], errors="coerce")
        duplicated = relevant.duplicated(KEYS + ["event_type_normalized"], keep=False)
        if duplicated.any():
            sample = relevant.loc[duplicated, KEYS + ["event_type"]].head(6).to_dict("records")
            raise ValueError(f"{EVENT_FILE} has duplicate event types per trial: {sample}")
        wide = relevant.pivot(
            index=KEYS, columns="event_type_normalized", values="timestamp_s"
        ).reset_index()
        wide.columns.name = None
        wide = wide.rename(
            columns={
                "target_onset": "target_onset_time_s",
                "response_deadline": "response_deadline_time_s",
            }
        )
        if "target_onset_time_s" not in wide:
            wide["target_onset_time_s"] = np.nan
        if "response_deadline_time_s" not in wide:
            wide["response_deadline_time_s"] = np.nan
        source_columns = [c for c in ("event_source", "timing_source") if c in events]
        source_values = []
        for column in source_columns:
            source_values.extend(events[column].dropna().astype(str).str.strip().tolist())
        source_values = sorted({value for value in source_values if value})
        wide["timing_source"] = "; ".join(source_values) if source_values else "event_log.csv"
        return _normalize_keys(wide, EVENT_FILE), diagnostics

    if "target_onset_time_s" not in events:
        raise ValueError(
            f"{EVENT_FILE} needs target_onset_time_s (wide format), or event_type and "
            "timestamp_s (long format)"
        )
    diagnostics["format"] = "wide"
    for column in ("target_onset_time_s", "response_deadline_time_s"):
        if column not in events:
            events[column] = np.nan
        events[column] = pd.to_numeric(events[column], errors="coerce")
    if "timing_source" not in events:
        events["timing_source"] = ""
    return events[KEYS + ["target_onset_time_s", "response_deadline_time_s", "timing_source"]], diagnostics


def parse_response_header(label: Any) -> dict[str, Any]:
    """Read an L/R code legend embedded in a MAT DataLabel, when available."""
    text = "" if _blank(label) else str(label).strip()
    channel = text.split(":", maxsplit=1)[0].strip()
    match = re.search(
        r"L\s*([+-]?\d+(?:\.\d+)?)\s*/\s*R\s*([+-]?\d+(?:\.\d+)?)",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return {
            "response_channel_from_mat": channel or None,
            "response_label_from_mat": text or None,
            "mat_left_response_code": None,
            "mat_right_response_code": None,
            "mat_response_code_map_status": "unparsed_channel_label",
        }
    left_code, right_code = float(match.group(1)), float(match.group(2))
    if left_code == right_code:
        status = "invalid_identical_left_right_codes"
    else:
        status = "parsed_from_mat_channel_label"
    return {
        "response_channel_from_mat": channel,
        "response_label_from_mat": text,
        "mat_left_response_code": left_code,
        "mat_right_response_code": right_code,
        "mat_response_code_map_status": status,
    }


def _same_code(value: Any, expected: Any) -> bool:
    if _blank(value) or expected is None:
        return False
    try:
        return bool(np.isclose(float(value), float(expected), atol=1e-8, rtol=0.0))
    except (TypeError, ValueError):
        return False


def audit_record_metadata(
    trial_table: pd.DataFrame,
    mapping_table: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    mapping = _normalize_keys_for_records(mapping_table, MAPPING_FILE)
    if mapping["record"].duplicated().any():
        dup = mapping.loc[mapping["record"].duplicated(keep=False), "record"].tolist()
        raise ValueError(f"{MAPPING_FILE} has duplicate records: {dup}")
    mapping_by_record = mapping.set_index("record", drop=False).to_dict(orient="index")
    rows: list[dict[str, Any]] = []
    response_code_maps: dict[str, dict[str, Any]] = {}
    cue_codes_by_trial: dict[str, dict[int, Any]] = {}

    for record in sorted(trial_table["record"].astype(str).unique()):
        part = trial_table.loc[trial_table["record"].astype(str).eq(record)].copy()
        raw = load_raw_record(record)
        response_info = parse_response_header(raw.get("response_label"))
        meta = mapping_by_record.get(record, {})
        actual_channel = str(response_info.get("response_channel_from_mat") or "")
        declared_channel = "" if _blank(meta.get("response_channel")) else str(meta.get("response_channel")).strip()
        # response_channel is optional in the mapping table; if supplied, it is
        # checked against the MAT label. DataLabel remains the code-map source.
        channel_matches = (
            None if not declared_channel else declared_channel.casefold() == actual_channel.casefold()
        )

        left_code = response_info["mat_left_response_code"]
        right_code = response_info["mat_right_response_code"]
        observed_onset_codes = sorted(
            pd.to_numeric(part.get("response_raw", pd.Series(dtype=float)), errors="coerce")
            .dropna()
            .unique()
            .tolist()
        )
        observed_bout_codes = sorted({
            int(code)
            for sequence in part.get("response_bout_code_sequence", pd.Series(dtype=str)).dropna().astype(str)
            for code in sequence.split("|")
            if code.strip()
        })
        unmapped_bout_codes = [
            code
            for code in observed_bout_codes
            if left_code is None
            or right_code is None
            or (np.sign(code) != np.sign(left_code) and np.sign(code) != np.sign(right_code))
        ]
        onset_stage_codes = [
            code
            for code in observed_onset_codes
            if left_code is not None
            and right_code is not None
            and np.sign(code) in (np.sign(left_code), np.sign(right_code))
            and not (_same_code(code, left_code) or _same_code(code, right_code))
        ]
        response_map_ok = (
            response_info["mat_response_code_map_status"] == "parsed_from_mat_channel_label"
            and not unmapped_bout_codes
            and channel_matches is not False
        )
        declared_code_coverage = int(
            pd.to_numeric(part.get("response_declared_code", pd.Series(dtype=float)), errors="coerce")
            .notna().sum()
        )
        raw_response_signal = np.asarray(raw["response_signal"], dtype=float).reshape(-1)
        raw_response_counts = {
            str(float(code)): int(np.count_nonzero(raw_response_signal == code))
            for code in np.unique(raw_response_signal[np.isfinite(raw_response_signal)])
        }
        response_code_maps[record] = {
            "left_code": left_code,
            "right_code": right_code,
            "mapping_ok": bool(response_map_ok),
            "onset_stage_codes": onset_stage_codes,
            "unmapped_bout_codes": unmapped_bout_codes,
            "mapping_source": "raw MAT DataLabel",
        }

        cue_signal = np.asarray(raw["channels"]["VisCue"], dtype=float).reshape(-1)
        cue_info = parse_response_header(raw["labels"].get("VisCue"))
        cue_left_code = cue_info["mat_left_response_code"]
        cue_right_code = cue_info["mat_right_response_code"]
        cue_raw_map: dict[int, Any] = {}
        cue_observed_values: list[float] = []
        cue_side_mismatch = 0
        cue_code_outside_header = 0
        for row in part.itertuples(index=False):
            trial_index = int(row.original_trial_index)
            sample = int(getattr(row, "cue_sample_index"))
            if sample < 0 or sample >= cue_signal.size:
                cue_raw = np.nan
            else:
                cue_raw = float(cue_signal[sample])
            cue_raw_map[trial_index] = cue_raw
            if np.isfinite(cue_raw):
                cue_observed_values.append(cue_raw)
                if cue_left_code is not None and _same_code(cue_raw, cue_left_code):
                    expected_side = -1
                elif cue_right_code is not None and _same_code(cue_raw, cue_right_code):
                    expected_side = 1
                elif cue_left_code is not None and cue_right_code is not None:
                    cue_code_outside_header += 1
                    expected_side = int(np.sign(cue_raw))
                else:
                    expected_side = int(np.sign(cue_raw))
                existing_side = getattr(row, "cue_side", np.nan)
                if not _blank(existing_side) and int(float(existing_side)) != expected_side:
                    cue_side_mismatch += 1
        cue_codes_by_trial[record] = cue_raw_map
        task_candidate = record.rsplit("_", 1)[-1] if "_" in record else ""
        mapping_fields = ["participant_id", "task_id", "session_id", "mapping_source"]
        missing_identity = [field for field in mapping_fields if _blank(meta.get(field))]
        rows.append(
            {
                "record": record,
                "filename_task_suffix_candidate_only": task_candidate,
                "participant_id": meta.get("participant_id"),
                "task_id": meta.get("task_id"),
                "session_id": meta.get("session_id"),
                "mapping_source": meta.get("mapping_source"),
                "identity_mapping_status": "complete" if not missing_identity else "missing:" + ";".join(missing_identity),
                "viscue_channel_from_mat": raw["labels"].get("VisCue"),
                "viscue_event_raw_codes": json.dumps(
                    {str(float(k)): int(v) for k, v in pd.Series(cue_observed_values).value_counts().sort_index().items()},
                    ensure_ascii=False,
                ),
                "viscue_label_from_mat": cue_info["response_label_from_mat"],
                "mat_left_viscue_code": cue_left_code,
                "mat_right_viscue_code": cue_right_code,
                "viscue_code_map_status": (
                    "matched_to_mat_channel_label"
                    if cue_info["mat_response_code_map_status"] == "parsed_from_mat_channel_label"
                    and cue_code_outside_header == 0
                    else cue_info["mat_response_code_map_status"]
                ),
                "viscue_side_rule_in_current_pipeline": "MAT L/R legend when parsed; otherwise event sign",
                "viscue_event_codes_outside_mat_legend": int(cue_code_outside_header),
                "viscue_code_side_mismatches_with_trial_table": int(cue_side_mismatch),
                **response_info,
                "declared_response_channel": declared_channel or None,
                "response_channel_matches_mat": channel_matches,
                "observed_response_event_codes": json.dumps(observed_onset_codes),
                "observed_response_bout_codes": json.dumps(observed_bout_codes),
                "response_declared_code_in_bout_count": declared_code_coverage,
                "raw_response_sample_code_counts": json.dumps(raw_response_counts),
                "response_onset_stage_codes_not_in_DataLabel": json.dumps(onset_stage_codes),
                "unmapped_response_bout_codes": json.dumps(unmapped_bout_codes),
                "response_code_map_status": (
                    "matched"
                    if response_map_ok and not onset_stage_codes
                    else "declared_code_in_bout_with_onset_stage_codes"
                    if response_map_ok and onset_stage_codes
                    else response_info["mat_response_code_map_status"]
                ),
            }
        )
    return pd.DataFrame(rows), response_code_maps, cue_codes_by_trial


def _normalize_keys_for_records(frame: pd.DataFrame, source_name: str) -> pd.DataFrame:
    if "record" not in frame.columns:
        raise ValueError(f"{source_name} needs a record column")
    out = frame.copy()
    out["record"] = out["record"].astype("string").str.strip()
    required = ["participant_id", "task_id", "session_id", "mapping_source"]
    for column in required:
        if column not in out:
            out[column] = ""
    if "response_channel" not in out:
        out["response_channel"] = ""
    return out


def _record_end_times(records: list[str]) -> dict[str, float]:
    result: dict[str, float] = {}
    for record in records:
        raw = load_raw_record(record)
        times = np.asarray(raw["channels"]["TimeStamp"], dtype=float).reshape(-1)
        finite = times[np.isfinite(times)]
        result[record] = float(np.max(finite)) if finite.size else np.nan
    return result


def _map_response_side(
    raw_code: Any,
    response_event_count: Any,
    code_map: dict[str, Any],
) -> tuple[float, str]:
    count = pd.to_numeric(pd.Series([response_event_count]), errors="coerce").iloc[0]
    if pd.isna(count) or int(count) == 0:
        return np.nan, "no_response_marker"
    multiple = int(count) > 1
    if _blank(raw_code):
        return np.nan, "response_code_missing"
    if not code_map.get("mapping_ok", False):
        return np.nan, "response_code_map_unverified"
    if _same_code(raw_code, code_map.get("left_code")):
        return -1.0, "first_response_from_multiple_events" if multiple else "decoded_from_mat_channel_label"
    if _same_code(raw_code, code_map.get("right_code")):
        return 1.0, "first_response_from_multiple_events" if multiple else "decoded_from_mat_channel_label"
    if np.sign(float(raw_code)) == np.sign(float(code_map.get("left_code"))):
        return -1.0, "first_response_from_multiple_events" if multiple else "decoded_by_mat_legend_sign_magnitude_differs"
    if np.sign(float(raw_code)) == np.sign(float(code_map.get("right_code"))):
        return 1.0, "first_response_from_multiple_events" if multiple else "decoded_by_mat_legend_sign_magnitude_differs"
    return np.nan, "response_code_not_in_mat_channel_legend"


def _outcome_row(row: Any, code_map: dict[str, Any], record_end: float) -> dict[str, Any]:
    if hasattr(row, "response_decode_status"):
        response_side = pd.to_numeric(
            pd.Series([getattr(row, "choice_side", np.nan)]), errors="coerce"
        ).iloc[0]
        response_decode_status = str(getattr(row, "response_decode_status"))
    else:
        response_side, response_decode_status = _map_response_side(
            getattr(row, "response_raw", np.nan),
            getattr(row, "response_event_count", np.nan),
            code_map,
        )
    channel_target_side = pd.to_numeric(
        pd.Series([getattr(row, "cue_side", np.nan)]), errors="coerce"
    ).iloc[0]
    external_correct_side = pd.to_numeric(
        pd.Series([getattr(row, "correct_response_side_truth", np.nan)]),
        errors="coerce",
    ).iloc[0]
    if np.isfinite(channel_target_side) and np.isfinite(response_side):
        correctness = "correct" if response_side == channel_target_side else "incorrect"
    elif not np.isfinite(channel_target_side):
        correctness = "target_side_missing"
    elif response_decode_status == "no_response_marker":
        correctness = "no_response"
    else:
        correctness = "unknown"

    target_onset = pd.to_numeric(pd.Series([getattr(row, "target_onset_time_s", np.nan)]), errors="coerce").iloc[0]
    schedule_anchor = pd.to_numeric(pd.Series([getattr(row, "target_time_s", np.nan)]), errors="coerce").iloc[0]
    schedule_anchor_source = getattr(row, "target_time_source", "")
    deadline = pd.to_numeric(pd.Series([getattr(row, "response_deadline_time_s", np.nan)]), errors="coerce").iloc[0]
    response_time = pd.to_numeric(pd.Series([getattr(row, "response_time_s", np.nan)]), errors="coerce").iloc[0]
    event_count = pd.to_numeric(pd.Series([getattr(row, "response_event_count", np.nan)]), errors="coerce").iloc[0]
    authoritative_omission = getattr(row, "authoritative_omission", np.nan)

    if np.isfinite(response_time) and np.isfinite(target_onset):
        reaction_time = float(response_time - target_onset)
        reaction_time_status = "ok" if reaction_time >= 0 else "response_precedes_target_check_clock"
    else:
        reaction_time = np.nan
        reaction_time_status = "target_onset_missing" if not np.isfinite(target_onset) else "response_time_missing"

    if np.isfinite(response_time) and np.isfinite(schedule_anchor):
        schedule_reaction_time = float(response_time - schedule_anchor)
        schedule_reaction_status = (
            "schedule_assumption_only"
            if schedule_reaction_time >= 0
            else "response_precedes_schedule_anchor_review"
        )
    else:
        schedule_reaction_time = np.nan
        schedule_reaction_status = "schedule_anchor_missing" if not np.isfinite(schedule_anchor) else "response_time_missing"
    target_time_source = (
        "external_event_log"
        if np.isfinite(target_onset)
        else str(schedule_anchor_source or "")
    )

    has_response = bool(np.isfinite(event_count) and int(event_count) > 0)
    custom_timeliness_status = str(getattr(row, "timeliness_status", "")).strip()
    if not custom_timeliness_status or custom_timeliness_status.lower() == "nan":
        custom_timeliness_status = "custom_timeliness_not_available"
    custom_is_timely = pd.to_numeric(
        pd.Series([getattr(row, "is_timely", np.nan)]), errors="coerce"
    ).iloc[0]
    custom_is_late = pd.to_numeric(
        pd.Series([getattr(row, "is_late", np.nan)]), errors="coerce"
    ).iloc[0]
    response_duration = pd.to_numeric(
        pd.Series([getattr(row, "response_duration_s", np.nan)]), errors="coerce"
    ).iloc[0]
    omission_status = (
        "response_observed_in_cue_interval"
        if has_response
        else "no_channel9_action_in_cue_interval"
    )
    timing_status = custom_timeliness_status
    outcome = (
        f"{correctness}_timely"
        if correctness in {"correct", "incorrect"} and custom_is_timely == 1
        else f"{correctness}_late"
        if correctness in {"correct", "incorrect"} and custom_is_late == 1
        else correctness
        if correctness in {"correct", "incorrect"}
        else "no_response_in_cue_interval"
        if not has_response
        else "outcome_unknown"
    )

    return {
        "actual_response_side_from_mat": response_side,
        "response_decode_status": response_decode_status,
        "response_declared_code": getattr(row, "response_declared_code", np.nan),
        "correctness_status": correctness,
        "correct": int(correctness == "correct") if correctness in {"correct", "incorrect"} else np.nan,
        "is_omission": int(not has_response),
        "target_side_used": channel_target_side,
        "target_side_source": "channel8_VisCue",
        "external_correct_response_side": external_correct_side,
        "external_target_side_disagreement": bool(
            np.isfinite(external_correct_side)
            and np.isfinite(channel_target_side)
            and external_correct_side != channel_target_side
        ),
        "target_onset_time_s": target_onset,
        "schedule_target_anchor_time_s": schedule_anchor,
        "target_time_source": target_time_source,
        "response_deadline_time_s": deadline,
        "response_time_s": response_time,
        "response_analysis_window_start_s": getattr(row, "response_analysis_window_start_s", np.nan),
        "response_analysis_window_end_s": getattr(row, "response_analysis_window_end_s", np.nan),
        "response_present_in_analysis_window": getattr(row, "response_present_in_analysis_window", np.nan),
        "response_duration_s": response_duration,
        "timeliness_status": custom_timeliness_status,
        "is_timely": custom_is_timely,
        "is_late": custom_is_late,
        "reaction_time_s_from_verified_target": reaction_time,
        "reaction_time_status": reaction_time_status,
        "reaction_time_s_from_schedule_anchor": schedule_reaction_time,
        "schedule_anchor_reaction_time_status": schedule_reaction_status,
        "response_timing_status": timing_status,
        "record_end_time_s": record_end,
        "recording_covers_deadline": bool(np.isfinite(deadline) and np.isfinite(record_end) and record_end >= deadline),
        "omission_status": omission_status,
        "trial_outcome": outcome,
    }


def _key_coverage(base: pd.DataFrame, external: pd.DataFrame, label: str) -> dict[str, Any]:
    base_keys = pd.MultiIndex.from_frame(base[KEYS])
    ext_keys = pd.MultiIndex.from_frame(external[KEYS])
    matched = ext_keys.isin(base_keys)
    return {
        f"{label}_rows": int(len(external)),
        f"{label}_matched_rows": int(matched.sum()),
        f"{label}_unmatched_rows": int((~matched).sum()),
        f"{label}_base_trials_without_row": int((~base_keys.isin(ext_keys)).sum()),
    }


def analyze(
    trial_table_path: Path,
    input_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    trial_table = _normalize_keys(_read_csv(trial_table_path, "Q3 trial table"), "trial_table.csv")
    _assert_unique_keys(trial_table, "trial_table.csv")
    required_trial_columns = {
        "cue_side",
        "cue_sample_index",
        "response_raw",
        "response_event_count",
        "response_time_s",
        "response_channel_label",
    }
    missing_trial_columns = sorted(required_trial_columns.difference(trial_table.columns))
    if missing_trial_columns:
        raise ValueError(f"trial_table.csv is missing required fields: {missing_trial_columns}")

    truth = load_truth_table(input_dir / TRUTH_FILE)
    event_path = input_dir / EVENT_FILE
    if event_path.exists():
        event_times, event_diagnostics = load_event_table(event_path)
    else:
        event_times = trial_table[KEYS].copy()
        event_times["target_onset_time_s"] = np.nan
        event_times["response_deadline_time_s"] = np.nan
        event_times["timing_source"] = ""
        event_diagnostics = {"format": "not_supplied", "ignored_event_types": []}
    mapping_path = input_dir / MAPPING_FILE
    if mapping_path.exists():
        mapping_table = _read_csv(mapping_path, MAPPING_FILE)
    else:
        mapping_table = pd.DataFrame(
            {"record": sorted(trial_table["record"].astype(str).unique())}
        )
    metadata_audit, response_maps, cue_raw_codes = audit_record_metadata(trial_table, mapping_table)
    record_ends = _record_end_times(sorted(trial_table["record"].astype(str).unique()))

    coverage = {
        **_key_coverage(trial_table, truth, "truth"),
        **_key_coverage(trial_table, event_times, "event_log"),
    }
    merged = trial_table.merge(
        truth,
        on=KEYS,
        how="left",
        validate="one_to_one",
        suffixes=("", "_truth"),
    ).merge(
        event_times,
        on=KEYS,
        how="left",
        validate="one_to_one",
        suffixes=("", "_event"),
    )

    metadata_by_record = metadata_audit.set_index("record").to_dict(orient="index")
    outcome_rows: list[dict[str, Any]] = []
    for row in merged.itertuples(index=False):
        record = str(row.record)
        derived = _outcome_row(row, response_maps.get(record, {}), record_ends.get(record, np.nan))
        raw_cue = cue_raw_codes.get(record, {}).get(int(row.original_trial_index), np.nan)
        outcome_rows.append(
            {
                "record": record,
                "original_trial_index": int(row.original_trial_index),
                "cue_side_from_mat": getattr(row, "cue_side", np.nan),
                "viscue_raw_code_at_event": raw_cue,
                "external_truth_source": getattr(row, "truth_source_truth", getattr(row, "truth_source", "")),
                "authoritative_omission": getattr(row, "authoritative_omission_truth", getattr(row, "authoritative_omission", np.nan)),
                "response_event_count": getattr(row, "response_event_count", np.nan),
                "response_raw_code": getattr(row, "response_raw", np.nan),
                "response_bout_code_sequence": getattr(row, "response_bout_code_sequence", ""),
                "mat_response_channel_label": getattr(row, "response_channel_label", ""),
                **derived,
                "timing_source": getattr(row, "timing_source", ""),
                "identity_mapping_status": metadata_by_record.get(record, {}).get("identity_mapping_status", "missing_record_mapping"),
            }
        )
    outcomes = pd.DataFrame(outcome_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    outcomes.to_csv(output_dir / "behavior_trial_audit.csv", index=False, encoding="utf-8-sig")
    metadata_audit.to_csv(output_dir / "record_metadata_audit.csv", index=False, encoding="utf-8-sig")

    summaries: list[dict[str, Any]] = []
    for record, part in outcomes.groupby("record", sort=True):
        summaries.append(
            {
                "record": record,
                "n_trials": int(len(part)),
                "n_external_truth_labels": int(part["external_correct_response_side"].notna().sum()),
                "n_channel8_target_labels": int(part["target_side_used"].notna().sum()),
                "n_target_onsets": int(part["target_onset_time_s"].notna().sum()),
                "n_schedule_target_anchors": int(part["schedule_target_anchor_time_s"].notna().sum()),
                "n_deadlines": int(part["response_deadline_time_s"].notna().sum()),
                "n_response_markers": int(pd.to_numeric(part["response_event_count"], errors="coerce").fillna(0).gt(0).sum()),
                "n_decoded_response_sides": int(part["actual_response_side_from_mat"].notna().sum()),
                "n_correctness_known": int(part["correctness_status"].isin(["correct", "incorrect"]).sum()),
                "n_correct": int(part["correctness_status"].eq("correct").sum()),
                "n_incorrect": int(part["correctness_status"].eq("incorrect").sum()),
                "n_timely_by_cue_window": int(pd.to_numeric(part["is_timely"], errors="coerce").eq(1).sum()),
                "n_late_by_cue_window": int(pd.to_numeric(part["is_late"], errors="coerce").eq(1).sum()),
                "n_response_duration_labeled": int(pd.to_numeric(part["response_duration_s"], errors="coerce").notna().sum()),
                "n_omission_intervals": int(pd.to_numeric(part["is_omission"], errors="coerce").eq(1).sum()),
                "n_response_intervals": int(pd.to_numeric(part["is_omission"], errors="coerce").eq(0).sum()),
                "accuracy_among_decodable_responses": (
                    float(part.loc[part["correctness_status"].isin(["correct", "incorrect"]), "correctness_status"].eq("correct").mean())
                    if part["correctness_status"].isin(["correct", "incorrect"]).any()
                    else None
                ),
            }
        )
    summary_by_record = pd.DataFrame(summaries)
    summary_by_record.to_csv(output_dir / "behavior_record_summary.csv", index=False, encoding="utf-8-sig")

    field_coverage = {
        "trials_with_external_correct_response_side": int(
            pd.to_numeric(merged["correct_response_side_truth"], errors="coerce").notna().sum()
        ),
        "trials_with_external_truth_source": int(
            merged["truth_source_truth"].fillna("").astype(str).str.strip().ne("").sum()
            if "truth_source_truth" in merged
            else merged["truth_source"].fillna("").astype(str).str.strip().ne("").sum()
            if "truth_source" in merged
            else 0
        ),
        "trials_with_channel8_target_side": int(outcomes["target_side_used"].notna().sum()),
        "trials_with_channel8_9_correctness": int(outcomes["correct"].notna().sum()),
        "trials_with_interval_omission_label": int(outcomes["is_omission"].notna().sum()),
        "trials_with_cue_window_timeliness_label": int(pd.to_numeric(outcomes["is_timely"], errors="coerce").notna().sum()),
        "trials_with_response_duration": int(pd.to_numeric(outcomes["response_duration_s"], errors="coerce").notna().sum()),
        "trials_with_target_onset": int(pd.to_numeric(merged["target_onset_time_s"], errors="coerce").notna().sum()),
        "trials_with_response_deadline": int(pd.to_numeric(merged["response_deadline_time_s"], errors="coerce").notna().sum()),
        "records_with_complete_identity_mapping": int(metadata_audit["identity_mapping_status"].eq("complete").sum()),
    }
    fully_observed = (
        field_coverage["trials_with_target_onset"] == len(merged)
        and field_coverage["trials_with_response_deadline"] == len(merged)
        and field_coverage["records_with_complete_identity_mapping"] == len(metadata_audit)
    )
    summary = {
        "status": "external_timing_or_identity_metadata_complete" if fully_observed else "channel_outcomes_derived_external_timing_metadata_partial",
        "analysis_unit": "trial keyed by record and original_trial_index",
        "trial_table": str(trial_table_path),
        "input_files": {
            "trial_truth": str(input_dir / TRUTH_FILE),
            "event_log": str(input_dir / EVENT_FILE),
            "record_mapping": str(input_dir / MAPPING_FILE),
        },
        "n_trial_rows": int(len(trial_table)),
        "n_records": int(trial_table["record"].nunique()),
        "external_key_coverage": coverage,
        "external_field_coverage": field_coverage,
        "derived_outcome_counts": {
            "correct": int(pd.to_numeric(outcomes["correct"], errors="coerce").eq(1).sum()),
            "incorrect": int(pd.to_numeric(outcomes["correct"], errors="coerce").eq(0).sum()),
            "omission_in_cue_interval": int(pd.to_numeric(outcomes["is_omission"], errors="coerce").eq(1).sum()),
            "timely_by_cue_window": int(pd.to_numeric(outcomes["is_timely"], errors="coerce").eq(1).sum()),
            "late_by_cue_window": int(pd.to_numeric(outcomes["is_late"], errors="coerce").eq(1).sum()),
        },
        "event_log_format": event_diagnostics,
        "record_metadata_complete": int(metadata_audit["identity_mapping_status"].eq("complete").sum()),
        "record_metadata_total": int(len(metadata_audit)),
        "correctness_rule": "compare channel-8 VisCue target side with the DataLabel-declared left/right code found in the same contiguous channel-9 nonzero bout; preserve the first 0-to-nonzero edge as response onset; no channel-9 bout in a cue interval is an operational omission",
        "omission_rule": "no channel-9 action bout in the cue-onset-to-next-cue interval",
        "timeliness_window_s_relative_to_channel8_cue": [-1.0, 5.0],
        "timeliness_rule": "timely when the channel-9 L/R code declared in DataLabel is observed within cue-1 to cue+5 s; Task-1 Action codes are +/-1 and Task-2 TgtAct codes are +/-2; an action in the cue interval without its declared code in this window is late",
        "response_duration_rule": "duration of the first contiguous channel-9 nonzero bout (sample count / sample rate); this is not target-to-response reaction time",
        "late_response_status": "classified by the task-defined cue window; this is not an independently verified experiment deadline",
        "timing_rule": "external target onset/deadline are absolute seconds on the MAT TimeStamp clock; if actual target onset is absent, the existing cue+2.2 s schedule anchor is reported separately as an assumption",
        "coding_rule": "VisCue event sign is reported from raw samples; Action/TgtAct left/right codes are parsed from the MAT DataLabel",
        "response_onset_stage_code_records": metadata_audit.loc[
            metadata_audit["response_code_map_status"].eq("declared_code_in_bout_with_onset_stage_codes"), "record"
        ].astype(str).tolist(),
        "outputs": [
            "behavior_trial_audit.csv",
            "behavior_record_summary.csv",
            "record_metadata_audit.csv",
            "behavior_event_audit_summary.json",
        ],
    }
    (output_dir / "behavior_event_audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    return summary


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if pd.isna(value):
        return None
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial-table", type=Path, default=OUTPUT_DIR / "trial_table.csv")
    parser.add_argument("--input-dir", type=Path, default=INPUT_DIR)
    parser.add_argument(
        "--output-dir", type=Path, default=OUTPUT_DIR / "behavior_event_audit"
    )
    args = parser.parse_args()

    if not args.trial_table.exists():
        raise FileNotFoundError(
            f"{args.trial_table} is missing; run 01_audit_events.py and "
            "02_extract_trials.py first"
        )
    trial_table = _normalize_keys(_read_csv(args.trial_table, "Q3 trial table"), "trial_table.csv")
    created = create_missing_input_templates(args.input_dir, trial_table)
    required_paths = [
        args.input_dir / TRUTH_FILE,
        args.input_dir / EVENT_FILE,
        args.input_dir / MAPPING_FILE,
    ]
    missing = [path for path in required_paths if not path.exists()]
    if missing:
        print("Created input templates:")
        for path in created:
            print(f"  {path}")
        print("The audit can derive correctness and cue-interval omissions from channels 8/9; external files add timing and record metadata only.")

    summary = analyze(args.trial_table, args.input_dir, args.output_dir)
    if created:
        print("Created input templates:")
        for path in created:
            print(f"  {path}")
    print(f"Wrote behavior audit for {summary['n_trial_rows']} trials to {args.output_dir}")
    print(
        "Verified fields: "
        f"channel-derived correctness labels={summary['external_field_coverage']['trials_with_channel8_9_correctness']}/"
        f"{summary['n_trial_rows']}, external truth labels="
        f"{summary['external_field_coverage']['trials_with_external_correct_response_side']}/"
        f"{summary['n_trial_rows']}, target onsets="
        f"{summary['external_field_coverage']['trials_with_target_onset']}/"
        f"{summary['n_trial_rows']}, deadlines="
        f"{summary['external_field_coverage']['trials_with_response_deadline']}/"
        f"{summary['n_trial_rows']}, record mappings="
        f"{summary['external_field_coverage']['records_with_complete_identity_mapping']}/"
        f"{summary['record_metadata_total']}"
    )


if __name__ == "__main__":
    main()
