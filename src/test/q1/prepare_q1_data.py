from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


MISSING_TOKENS = {"/////", "//////", "////////", "///", "-", ""}
DATA_TYPE_NAMES = {
    11: "temperature_c",
    12: "water_vapor_density",
    13: "relative_humidity",
    14: "liquid_water",
}


def first_column_with_prefix(columns: pd.Index, prefix: str) -> str | None:
    for column in columns:
        if str(column).startswith(prefix):
            return str(column)
    return None


@dataclass(frozen=True)
class StationSpec:
    station: str
    root: Path
    microwave_file: Path
    wind_profiler_dir: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare model-ready data for problem D question 1."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(r"D:\8\Desktop\CMathc\data\D题\题目数据及检验\随题数据\第一题"),
        help="Directory containing question 1 station folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
        help="Directory for cleaned outputs.",
    )
    return parser.parse_args()


def to_float(value: str) -> float:
    value = value.strip()
    if value in MISSING_TOKENS or set(value) == {"/"}:
        return math.nan
    try:
        parsed = float(value)
    except ValueError:
        return math.nan
    if parsed in {9999.0, 99999.0, 999999.0, 999998.0, -9999.0}:
        return math.nan
    return parsed


def find_station_specs(data_root: Path) -> list[StationSpec]:
    specs: list[StationSpec] = []
    for station_name in ("a站点", "b站点"):
        station_root = data_root / station_name
        microwave_files = sorted(station_root.glob("*微波辐射计数据.txt"))
        wind_dirs = [p for p in station_root.iterdir() if p.is_dir() and "风廓线雷达" in p.name]
        if not microwave_files:
            raise FileNotFoundError(f"No microwave radiometer file under {station_root}")
        if not wind_dirs:
            raise FileNotFoundError(f"No wind profiler directory under {station_root}")
        specs.append(
            StationSpec(
                station=station_name[0],
                root=station_root,
                microwave_file=microwave_files[0],
                wind_profiler_dir=wind_dirs[0],
            )
        )
    return specs


def read_microwave(path: Path, station: str) -> tuple[pd.DataFrame, dict[str, object]]:
    lines = path.read_text(encoding="gb18030", errors="replace").splitlines()
    rows = [line.split("\t") for line in lines if line.strip()]
    if len(rows) < 3:
        raise ValueError(f"Microwave file is too short: {path}")
    location_row = [cell for cell in rows[0] if cell.strip()]
    header = rows[1]
    width = len(header)
    body = [row[:width] + [""] * max(0, width - len(row)) for row in rows[2:]]
    data = pd.DataFrame(body, columns=header)
    data = data.loc[:, [c for c in data.columns if c and c != "nan"]]

    height_cols = [c for c in data.columns if c.endswith("(km)") and re.match(r"^\d", c)]
    id_cols = [c for c in data.columns if c not in height_cols]
    long = data.melt(id_vars=id_cols, value_vars=height_cols, var_name="height_label", value_name="value")
    long["height_m"] = long["height_label"].str.extract(r"([0-9.]+)").astype(float) * 1000.0
    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    long["data_type"] = pd.to_numeric(long["10"], errors="coerce").astype("Int64")
    long["variable"] = long["data_type"].map(DATA_TYPE_NAMES).fillna("unknown")
    long["time"] = pd.to_datetime(long["DateTime"], errors="coerce")
    long["station"] = station
    surface_temp_col = first_column_with_prefix(long.columns, "SurTem")
    surface_humidity_col = first_column_with_prefix(long.columns, "SurHum")
    surface_pressure_col = first_column_with_prefix(long.columns, "SurPre")
    for col in (surface_temp_col, surface_humidity_col, surface_pressure_col, "QCflag"):
        if col and col in long.columns:
            long[col] = pd.to_numeric(long[col], errors="coerce")

    profile = (
        long.pivot_table(
            index=["station", "time", "height_m"],
            columns="variable",
            values="value",
            aggfunc="mean",
        )
        .reset_index()
        .sort_values(["station", "time", "height_m"])
    )

    if surface_pressure_col:
        surface_pressure = (
            long.groupby(["station", "time"], as_index=False)[surface_pressure_col]
            .mean()
            .rename(columns={surface_pressure_col: "surface_pressure_hpa"})
        )
        profile = profile.merge(surface_pressure, on=["station", "time"], how="left")
    if surface_temp_col:
        surface_temp = (
            long.groupby(["station", "time"], as_index=False)[surface_temp_col]
            .mean()
            .rename(columns={surface_temp_col: "surface_temperature_c"})
        )
        profile = profile.merge(surface_temp, on=["station", "time"], how="left")
    if surface_humidity_col:
        surface_humidity = (
            long.groupby(["station", "time"], as_index=False)[surface_humidity_col]
            .mean()
            .rename(columns={surface_humidity_col: "surface_relative_humidity"})
        )
        profile = profile.merge(surface_humidity, on=["station", "time"], how="left")
    profile["pressure_hpa_est"] = profile["surface_pressure_hpa"] * (
        1.0 - 2.25577e-5 * profile["height_m"]
    ).clip(lower=0.1) ** 5.25588
    if "temperature_c" in profile.columns:
        temp_k = profile["temperature_c"] + 273.15
        profile["potential_temperature_k"] = temp_k * (1000.0 / profile["pressure_hpa_est"]) ** 0.286

    summary = {
        "file": str(path),
        "location_row": location_row,
        "rows": int(len(profile)),
        "times": int(profile["time"].nunique()),
        "height_min_m": float(profile["height_m"].min()),
        "height_max_m": float(profile["height_m"].max()),
        "variables": sorted(profile.columns.difference(["station", "time", "height_m"]).tolist()),
    }
    return profile, summary


def parse_timestamp_from_name(path: Path) -> pd.Timestamp:
    match = re.search(r"(20\d{12})", path.name)
    if not match:
        raise ValueError(f"No timestamp in filename: {path.name}")
    return pd.to_datetime(match.group(1), format="%Y%m%d%H%M%S")


def read_robs(path: Path, station: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    time = parse_timestamp_from_name(path)
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        in_block = False
        for line in fh:
            line = line.strip()
            if line == "ROBS":
                in_block = True
                continue
            if line == "NNNN":
                break
            if not in_block or not line:
                continue
            parts = line.split()
            if len(parts) < 7:
                continue
            rows.append(
                {
                    "station": station,
                    "time": time,
                    "height_m": to_float(parts[0]),
                    "wind_direction_deg": to_float(parts[1]),
                    "wind_speed_ms": to_float(parts[2]),
                    "vertical_velocity_ms": to_float(parts[3]),
                    "horizontal_reliability": to_float(parts[4]),
                    "vertical_reliability": to_float(parts[5]),
                    "cn2": to_float(parts[6]),
                    "source_file": path.name,
                }
            )
    return pd.DataFrame(rows)


def read_rad(path: Path, station: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    time = parse_timestamp_from_name(path)
    beam: str | None = None
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if line.startswith("RAD "):
                beam = line.removeprefix("RAD ").strip().lower().replace(" ", "_")
                continue
            if line == "NNNN":
                beam = None
                continue
            if beam is None or not line:
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            rows.append(
                {
                    "station": station,
                    "time": time,
                    "beam": beam,
                    "height_m": to_float(parts[0]),
                    "rad_col1": to_float(parts[1]),
                    "rad_col2": to_float(parts[2]),
                    "rad_col3": to_float(parts[3]),
                    "source_file": path.name,
                }
            )
    data = pd.DataFrame(rows)
    if data.empty:
        return data
    numeric_cols = ["rad_col1", "rad_col2", "rad_col3"]
    wide = (
        data.groupby(["station", "time", "height_m"], as_index=False)[numeric_cols]
        .agg(["mean", "std", "count"])
    )
    wide.columns = [
        "_".join(col).strip("_") if isinstance(col, tuple) else col for col in wide.columns
    ]
    return wide.rename(
        columns={
            "rad_col1_mean": "rad_feature1_mean",
            "rad_col2_mean": "rad_feature2_mean",
            "rad_col3_mean": "rad_feature3_mean",
            "rad_col1_std": "rad_feature1_std",
            "rad_col2_std": "rad_feature2_std",
            "rad_col3_std": "rad_feature3_std",
            "rad_col1_count": "rad_feature1_count",
            "rad_col2_count": "rad_feature2_count",
            "rad_col3_count": "rad_feature3_count",
        }
    )


def read_wind_profiler(spec: StationSpec) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    robs_frames = [read_robs(p, spec.station) for p in sorted(spec.wind_profiler_dir.glob("*ROBS.TXT"))]
    rad_frames = [read_rad(p, spec.station) for p in sorted(spec.wind_profiler_dir.glob("*RAD.TXT"))]
    robs = pd.concat(robs_frames, ignore_index=True) if robs_frames else pd.DataFrame()
    rad = pd.concat(rad_frames, ignore_index=True) if rad_frames else pd.DataFrame()
    summary = {
        "station": spec.station,
        "wind_profiler_dir": str(spec.wind_profiler_dir),
        "robs_files": len(robs_frames),
        "rad_files": len(rad_frames),
        "robs_rows": int(len(robs)),
        "rad_rows": int(len(rad)),
        "times": int(robs["time"].nunique()) if not robs.empty else 0,
    }
    return robs, rad, summary


def add_wind_components(data: pd.DataFrame) -> pd.DataFrame:
    out = data.copy()
    direction_rad = np.deg2rad(out["wind_direction_deg"])
    # Meteorological wind direction is the direction wind comes from.
    out["u_ms"] = -out["wind_speed_ms"] * np.sin(direction_rad)
    out["v_ms"] = -out["wind_speed_ms"] * np.cos(direction_rad)
    return out


def add_vertical_gradients(data: pd.DataFrame) -> pd.DataFrame:
    out = data.sort_values(["station", "time", "height_m"]).copy()
    for col in ["u_ms", "v_ms", "wind_speed_ms", "potential_temperature_k"]:
        if col in out.columns:
            out[f"d_{col}_dz"] = out.groupby(["station", "time"], group_keys=False).apply(
                lambda g: pd.Series(
                    np.gradient(g[col].to_numpy(dtype=float), g["height_m"].to_numpy(dtype=float)),
                    index=g.index,
                )
                if len(g) >= 2 and g[col].notna().sum() >= 2
                else pd.Series(np.nan, index=g.index),
                include_groups=False,
            )
    if {"d_u_ms_dz", "d_v_ms_dz"}.issubset(out.columns):
        out["vertical_shear_s2"] = out["d_u_ms_dz"] ** 2 + out["d_v_ms_dz"] ** 2
    if {"potential_temperature_k", "d_potential_temperature_k_dz", "vertical_shear_s2"}.issubset(
        out.columns
    ):
        out["richardson_number"] = (
            9.80665
            / out["potential_temperature_k"]
            * out["d_potential_temperature_k_dz"]
            / out["vertical_shear_s2"].replace(0, np.nan)
        )
    return out


def align_profiles(wind: pd.DataFrame, microwave: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    microwave = microwave.sort_values(["station", "time", "height_m"])
    thermal_cols = [
        col
        for col in [
            "temperature_c",
            "relative_humidity",
            "water_vapor_density",
            "liquid_water",
            "surface_pressure_hpa",
            "potential_temperature_k",
        ]
        if col in microwave.columns
    ]
    for (station, time), group in wind.groupby(["station", "time"], sort=True):
        mw_times = microwave.loc[microwave["station"] == station, "time"].dropna().drop_duplicates()
        if mw_times.empty:
            matched = group.copy()
            for col in thermal_cols:
                matched[col] = np.nan
            frames.append(matched)
            continue
        nearest_time = mw_times.iloc[np.argmin(np.abs((mw_times - time).dt.total_seconds()))]
        mw = microwave[(microwave["station"] == station) & (microwave["time"] == nearest_time)]
        matched = group.copy()
        matched["microwave_time"] = nearest_time
        matched["microwave_time_diff_min"] = abs((nearest_time - time).total_seconds()) / 60.0
        for col in thermal_cols:
            valid = mw[["height_m", col]].dropna()
            if len(valid) >= 2:
                matched[col] = np.interp(
                    matched["height_m"],
                    valid["height_m"],
                    valid[col],
                    left=np.nan,
                    right=np.nan,
                )
            else:
                matched[col] = np.nan
        frames.append(matched)
    return pd.concat(frames, ignore_index=True)


def build_outputs(data_root: Path, output_dir: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    station_specs = find_station_specs(data_root)

    microwave_frames: list[pd.DataFrame] = []
    robs_frames: list[pd.DataFrame] = []
    rad_frames: list[pd.DataFrame] = []
    summaries: dict[str, object] = {"stations": {}, "raw_data_root": str(data_root)}

    for spec in station_specs:
        microwave, microwave_summary = read_microwave(spec.microwave_file, spec.station)
        robs, rad, wind_summary = read_wind_profiler(spec)
        microwave_frames.append(microwave)
        robs_frames.append(robs)
        rad_frames.append(rad)
        summaries["stations"][spec.station] = {
            "microwave": microwave_summary,
            "wind_profiler": wind_summary,
        }

    microwave_all = pd.concat(microwave_frames, ignore_index=True)
    robs_all = pd.concat(robs_frames, ignore_index=True)
    rad_all = pd.concat(rad_frames, ignore_index=True)
    wind = robs_all.merge(rad_all, on=["station", "time", "height_m"], how="left")
    wind = add_wind_components(wind)
    model_data = align_profiles(wind, microwave_all)
    model_data = add_vertical_gradients(model_data)

    if "rad_feature1_mean" in model_data.columns:
        model_data["tke_proxy_from_radar"] = 0.5 * (
            model_data["rad_feature1_mean"] ** 2 + model_data["vertical_velocity_ms"] ** 2
        )
    if {"vertical_shear_s2", "d_potential_temperature_k_dz"}.issubset(model_data.columns):
        model_data["model_a_physical_proxy"] = model_data["vertical_shear_s2"] - (
            9.80665
            / model_data["potential_temperature_k"]
            * model_data["d_potential_temperature_k_dz"]
        )

    for frame in (microwave_all, robs_all, rad_all, model_data):
        for col in frame.select_dtypes(include=["datetime64[ns]"]).columns:
            frame[col] = frame[col].dt.strftime("%Y-%m-%d %H:%M:%S")

    outputs = {
        "microwave_profiles_csv": output_dir / "q1_microwave_profiles.csv",
        "wind_robs_csv": output_dir / "q1_wind_robs.csv",
        "wind_rad_features_csv": output_dir / "q1_wind_rad_features.csv",
        "model_dataset_csv": output_dir / "q1_model_dataset.csv",
        "summary_json": output_dir / "q1_summary.json",
    }
    microwave_all.to_csv(outputs["microwave_profiles_csv"], index=False, encoding="utf-8-sig")
    robs_all.to_csv(outputs["wind_robs_csv"], index=False, encoding="utf-8-sig")
    rad_all.to_csv(outputs["wind_rad_features_csv"], index=False, encoding="utf-8-sig")
    model_data.to_csv(outputs["model_dataset_csv"], index=False, encoding="utf-8-sig")

    summaries["outputs"] = {key: str(value) for key, value in outputs.items()}
    summaries["model_dataset"] = {
        "rows": int(len(model_data)),
        "columns": list(model_data.columns),
        "stations": sorted(model_data["station"].dropna().unique().tolist()),
        "missing_rate": {
            col: float(model_data[col].isna().mean())
            for col in model_data.columns
            if model_data[col].isna().any()
        },
        "role_notes": {
            "model_a": "Use thermal variables from microwave radiometer plus wind-profiler shear features.",
            "model_b": "Use only wind-profiler columns: wind, reliability, cn2, and RAD-derived features.",
            "leakage_guard": "No later question data or validation data is read by this script.",
        },
    }
    outputs["summary_json"].write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summaries


def main() -> None:
    args = parse_args()
    summary = build_outputs(args.data_root, args.output_dir)
    print(json.dumps(summary["model_dataset"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
