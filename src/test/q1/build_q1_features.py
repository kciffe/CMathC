from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
INPUT = BASE_DIR / "output" / "q1_dataset.csv"
OUTPUT = BASE_DIR / "output" / "q1_dataset_features.csv"


def add_features(df):
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"])

    # ????-??-??????? beam/mode????????????
    keep = [
        "station", "time", "height",
        "wind_direction", "wind_speed", "vertical_velocity", "cn2",
        "temperature", "relative_humidity", "water_vapor_density", "liquid_water",
    ]
    df = df[keep].groupby(["station", "time", "height"], as_index=False).mean()

    angle = np.deg2rad(df["wind_direction"])
    df["u"] = -df["wind_speed"] * np.sin(angle)
    df["v"] = -df["wind_speed"] * np.cos(angle)

    temp_k = df["temperature"] + 273.15
    pressure = 1013.25 * (1 - 0.0065 * df["height"] / 288.15) ** 5.255
    df["theta"] = temp_k * (1000 / pressure) ** 0.286

    result = []
    for _, g in df.groupby(["station", "time"]):
        g = g.sort_values("height").copy()
        g[["u", "v", "theta"]] = g[["u", "v", "theta"]].interpolate(limit_direction="both")

        z = g["height"].to_numpy()
        du_dz = np.gradient(g["u"].to_numpy(), z)
        dv_dz = np.gradient(g["v"].to_numpy(), z)
        dtheta_dz = np.gradient(g["theta"].to_numpy(), z)

        g["wind_shear"] = np.sqrt(du_dz**2 + dv_dz**2)
        g["n2"] = 9.80665 / g["theta"] * dtheta_dz
        g["ri"] = g["n2"] / (g["wind_shear"]**2 + 1e-6)
        result.append(g)

    return pd.concat(result, ignore_index=True)


if __name__ == "__main__":
    df = pd.read_csv(INPUT)
    out = add_features(df)
    out.to_csv(OUTPUT, index=False, encoding="utf-8-sig")
    print(f"saved: {OUTPUT}")
    print(out[["station", "time", "height", "wind_shear", "n2", "ri"]].head())
