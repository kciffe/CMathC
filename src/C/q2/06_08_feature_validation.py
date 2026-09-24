# -*- coding: utf-8 -*-
"""Validate mechanism-guided features on real, single-trial EEG records.

The classifier uses only observable EEG features. Every split holds out one
whole recording, and every feature is extracted independently from one trial.
"""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import trapezoid
from scipy.io import loadmat
from scipy.signal import savgol_filter, welch
from scipy.stats import ttest_ind
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_DIR = SCRIPT_DIR.parent / "q1" / "output" / "8riemann_denoise"
OUTPUT_DIR = SCRIPT_DIR / "output" / "06_08_feature_validation"

DATASETS = {
    "Task1": ["VisualCogA_Task-1", "VisualCogA_Task-2"],
    "Task2": ["VisualCogB_Task-1", "VisualCogB_Task-2"],
}

CHANNEL_NAMES = ["F3", "Fz", "F4"]
BASELINE = (-0.2, 0.0)
LATE_WINDOW = (0.25, 0.50)
SPECTRAL_WINDOW = (0.0, 0.80)
ROBUST_OUTLIER_Z = 10.0

FEATURES_ERP = [
    "F3晚期均值",
    "Fz晚期均值",
    "F4晚期均值",
    "Fz晚期峰潜伏期_ms",
]
FEATURES_SPATIAL = [
    "F4-F3晚期差",
    "Fz-(F3+F4)/2晚期差",
]
FEATURES_SPECTRAL = [
    "theta左右电极_log功率比(F4/F3)",
    "alpha左右电极_log功率比(F4/F3)",
    "beta左右电极_log功率比(F4/F3)",
]
ALL_FEATURES = FEATURES_ERP + FEATURES_SPATIAL + FEATURES_SPECTRAL

FEATURE_SETS = {
    "ERP": FEATURES_ERP,
    "ERP+空间": FEATURES_ERP + FEATURES_SPATIAL,
    "ERP+空间+频谱": ALL_FEATURES,
}
CLASSIFIERS = ("ShrinkageLDA", "LinearSVM")

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def unwrap_matlab_string(value):
    """Convert a MATLAB cell/string scalar into a Python string."""
    while isinstance(value, np.ndarray) and value.size == 1:
        value = value.reshape(-1)[0]
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").strip()
    return str(value).strip()


def baseline_correct(eeg, time_s):
    mask = (time_s >= BASELINE[0]) & (time_s < BASELINE[1])
    if not np.any(mask):
        raise ValueError(f"基线窗口 {BASELINE} 秒内没有采样点")
    baseline = eeg[:, mask].mean(axis=1, keepdims=True)
    return eeg - baseline


def band_power(signal, sample_rate, low_hz, high_hz):
    signal = np.asarray(signal, dtype=float)
    if signal.size < 2:
        raise ValueError("频谱窗口内少于两个采样点")
    signal = signal - np.mean(signal)
    nperseg = min(signal.size, int(round(sample_rate)))
    frequencies, psd = welch(signal, fs=sample_rate, nperseg=nperseg)
    band_mask = (frequencies >= low_hz) & (frequencies <= high_hz)
    if np.count_nonzero(band_mask) < 2:
        raise ValueError(
            f"{low_hz}–{high_hz} Hz 频带内 Welch 频点不足，无法积分功率"
        )
    return float(trapezoid(psd[band_mask], frequencies[band_mask]))


def peak_latency_ms(signal, time_s):
    """Return the smoothed positive peak latency within the predefined window."""
    if signal.size >= 7:
        signal = savgol_filter(signal, window_length=7, polyorder=2)
    return float(time_s[np.argmax(signal)] * 1000.0)


def extract_trial_features(eeg, time_s, sample_rate):
    late_mask = (time_s >= LATE_WINDOW[0]) & (time_s <= LATE_WINDOW[1])
    spectral_mask = (time_s >= SPECTRAL_WINDOW[0]) & (time_s <= SPECTRAL_WINDOW[1])
    if np.count_nonzero(late_mask) < 2:
        raise ValueError(f"晚期窗口 {LATE_WINDOW} 秒内采样点不足")
    if np.count_nonzero(spectral_mask) < 2:
        raise ValueError(f"频谱窗口 {SPECTRAL_WINDOW} 秒内采样点不足")

    f3, fz, f4 = eeg
    f3_mean = float(f3[late_mask].mean())
    fz_mean = float(fz[late_mask].mean())
    f4_mean = float(f4[late_mask].mean())

    features = {
        "F3晚期均值": f3_mean,
        "Fz晚期均值": fz_mean,
        "F4晚期均值": f4_mean,
        "Fz晚期峰潜伏期_ms": peak_latency_ms(fz[late_mask], time_s[late_mask]),
        "F4-F3晚期差": f4_mean - f3_mean,
        "Fz-(F3+F4)/2晚期差": fz_mean - 0.5 * (f3_mean + f4_mean),
    }

    bands = {
        "theta左右电极_log功率比(F4/F3)": (4.0, 8.0),
        "alpha左右电极_log功率比(F4/F3)": (8.0, 13.0),
        "beta左右电极_log功率比(F4/F3)": (13.0, 30.0),
    }
    for name, (low_hz, high_hz) in bands.items():
        f3_power = band_power(f3[spectral_mask], sample_rate, low_hz, high_hz)
        f4_power = band_power(f4[spectral_mask], sample_rate, low_hz, high_hz)
        # Positive values mean greater band power at F4 than F3.
        features[name] = float(np.log(f4_power + 1e-12) - np.log(f3_power + 1e-12))

    return features


def load_dataset_features(task, dataset):
    path = INPUT_DIR / f"{dataset}_clean.mat"
    if not path.is_file():
        raise FileNotFoundError(f"找不到输入 MAT 文件：{path}")
    mat = loadmat(path)
    required = {"trial_data", "relative_time", "cue_type", "SampleRate", "DataLabel"}
    missing = sorted(required.difference(mat))
    if missing:
        raise ValueError(f"{dataset}: MAT 缺少字段 {missing}")

    trials = np.asarray(mat["trial_data"], dtype=float)
    times = np.asarray(mat["relative_time"], dtype=float)
    raw_labels = np.asarray(mat["cue_type"], dtype=float).reshape(-1)
    if trials.ndim != 3:
        raise ValueError(f"{dataset}: trial_data 应为 Trial×Channel×Time，实际 {trials.shape}")
    n_trials, n_channels, n_samples = trials.shape
    if times.shape != (n_trials, n_samples):
        raise ValueError(f"{dataset}: relative_time={times.shape} 与 trial_data 不匹配")
    if raw_labels.size != n_trials or not np.all(np.isfinite(raw_labels)):
        raise ValueError(f"{dataset}: cue_type 数量或数值无效")
    if not np.all(np.isin(raw_labels, (-1.0, 1.0))):
        invalid = np.unique(raw_labels[~np.isin(raw_labels, (-1.0, 1.0))]).tolist()
        raise ValueError(f"{dataset}: cue_type 只能为 -1/+1，发现 {invalid}")
    labels = raw_labels.astype(int)
    label_counts = {label: int(np.count_nonzero(labels == label)) for label in (-1, 1)}
    if min(label_counts.values()) < 2:
        raise ValueError(f"{dataset}: 左右条件均需至少 2 个 Trial，当前 {label_counts}")

    if not np.all(np.isfinite(times)):
        raise ValueError(f"{dataset}: relative_time 含 NaN/Inf")
    if not np.all(np.diff(times[0]) > 0):
        raise ValueError(f"{dataset}: 时间轴必须严格递增")
    if not np.allclose(times, times[:1], rtol=0.0, atol=1e-12):
        raise ValueError(f"{dataset}: 各 Trial 的时间轴不一致")

    sample_rate_values = np.asarray(mat["SampleRate"], dtype=float).reshape(-1)
    if sample_rate_values.size != 1 or not np.isfinite(sample_rate_values[0]):
        raise ValueError(f"{dataset}: SampleRate 必须是单一有限数值")
    sample_rate = float(sample_rate_values[0])
    if sample_rate <= 0:
        raise ValueError(f"{dataset}: SampleRate 必须大于 0")

    data_labels = [
        unwrap_matlab_string(value)
        for value in np.asarray(mat["DataLabel"], dtype=object).reshape(-1)
    ]
    if len(data_labels) != n_channels:
        raise ValueError(
            f"{dataset}: DataLabel 有 {len(data_labels)} 项，trial_data 有 {n_channels} 个通道"
        )
    channel_indices = {}
    for channel in CHANNEL_NAMES:
        if data_labels.count(channel) != 1:
            raise ValueError(f"{dataset}: DataLabel 中应恰好包含一个 {channel}")
        channel_indices[channel] = data_labels.index(channel)

    time_s = times[0]
    for name, window in (
        ("baseline", BASELINE),
        ("late", LATE_WINDOW),
        ("spectral", SPECTRAL_WINDOW),
    ):
        if not np.any((time_s >= window[0]) & (time_s <= window[1])):
            raise ValueError(f"{dataset}: {name} 窗口 {window} 秒没有采样点")

    rows = []
    for trial_index in range(n_trials):
        eeg = trials[
            trial_index,
            [channel_indices[channel] for channel in CHANNEL_NAMES],
            :,
        ]
        eeg = baseline_correct(eeg, time_s)
        features = extract_trial_features(eeg, time_s, sample_rate)
        rows.append(
            {
                "Task": task,
                "Dataset": dataset,
                "Trial": trial_index,
                "Label": labels[trial_index],
                "方向": "左" if labels[trial_index] == -1 else "右",
                **features,
            }
        )
    return rows


def load_all_features():
    rows = []
    for task, datasets in DATASETS.items():
        for dataset in datasets:
            rows.extend(load_dataset_features(task, dataset))
    return pd.DataFrame(rows)


def audit_feature_quality(feature_df):
    """Report ranges and robust outlier candidates without modifying trials."""
    values = feature_df[ALL_FEATURES].to_numpy(dtype=float)
    if not np.all(np.isfinite(values)):
        locations = np.argwhere(~np.isfinite(values))[:10].tolist()
        raise ValueError(f"九个特征中发现 NaN/Inf，前十个位置：{locations}")
    if not np.all(np.isin(feature_df["Label"].to_numpy(), (-1, 1))):
        raise ValueError("单试次特征表的 Label 出现 -1/+1 以外的值")

    candidate_features = {index: [] for index in feature_df.index}
    print("\n特征质量核查（稳健 z>|10| 仅标记候选值，不删样本）：")
    for dataset, group in feature_df.groupby("Dataset", sort=False):
        left_n = int(np.count_nonzero(group["Label"].to_numpy() == -1))
        right_n = int(np.count_nonzero(group["Label"].to_numpy() == 1))
        print(f"{dataset}: Trial={len(group)}, 左={left_n}, 右={right_n}")
        for feature in ALL_FEATURES:
            x = group[feature].to_numpy(dtype=float)
            median = float(np.median(x))
            mad = float(np.median(np.abs(x - median)))
            unique_n = int(np.unique(x).size)
            if mad == 0.0:
                outlier_mask = x != median
            else:
                robust_z = 0.67448975 * (x - median) / mad
                outlier_mask = np.abs(robust_z) > ROBUST_OUTLIER_Z
            robust_outliers = int(np.count_nonzero(outlier_mask))
            for index in group.index[np.flatnonzero(outlier_mask)]:
                candidate_features[index].append(feature)
            print(
                f"  {feature}: unique={unique_n}, min={np.min(x):.6g}, "
                f"median={median:.6g}, max={np.max(x):.6g}, "
                f"robust_outlier_candidates={robust_outliers}"
            )
            if unique_n == 1:
                print(f"    警告：{feature} 在 {dataset} 内恒定")
            if robust_outliers:
                print("    提示：候选极端值保留在分析中；请结合原始波形/QC判断")

    # Keep the diagnostic discoverable in the trial-level CSV; these are audit
    # metadata columns and are never included in a model's feature matrix.
    feature_df["RobustOutlierCandidateCount"] = [
        len(candidate_features[index]) for index in feature_df.index
    ]
    feature_df["RobustOutlierCandidateFeatures"] = [
        ";".join(candidate_features[index]) for index in feature_df.index
    ]


def cohen_d(left, right):
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    n_left, n_right = len(left), len(right)
    pooled_variance = (
        (n_left - 1) * left.var(ddof=1) + (n_right - 1) * right.var(ddof=1)
    ) / (n_left + n_right - 2)
    pooled_sd = float(np.sqrt(pooled_variance))
    mean_difference = float(left.mean() - right.mean())
    if pooled_sd == 0.0:
        if mean_difference == 0.0:
            return 0.0
        return float(np.copysign(np.inf, mean_difference))
    return mean_difference / pooled_sd


def bh_fdr(p_values):
    p_values = np.asarray(p_values, dtype=float)
    if p_values.size == 0 or not np.all(np.isfinite(p_values)):
        raise ValueError("BH-FDR 输入的 p 值必须非空且有限")
    order = np.argsort(p_values)
    ranked = p_values[order]
    m = len(ranked)
    adjusted = ranked * m / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    result = np.empty_like(adjusted)
    result[order] = adjusted
    return result


def feature_statistics(feature_df):
    rows = []
    for dataset, group in feature_df.groupby("Dataset", sort=False):
        task = group["Task"].iloc[0]
        left_group = group.loc[group["Label"] == -1]
        right_group = group.loc[group["Label"] == 1]
        p_values = []
        dataset_rows = []

        for feature in ALL_FEATURES:
            left = left_group[feature].to_numpy(dtype=float)
            right = right_group[feature].to_numpy(dtype=float)
            test = ttest_ind(left, right, equal_var=False)
            if not np.isfinite(test.pvalue):
                raise ValueError(f"{dataset}/{feature}: Welch t 检验返回非有限 p 值")
            dataset_rows.append(
                {
                    "Task": task,
                    "Dataset": dataset,
                    "Feature": feature,
                    "左Trial数": len(left),
                    "右Trial数": len(right),
                    "左均值": float(left.mean()),
                    "右均值": float(right.mean()),
                    "均值差_左减右": float(left.mean() - right.mean()),
                    "Cohens_d": cohen_d(left, right),
                    "Welch_p": float(test.pvalue),
                }
            )
            p_values.append(float(test.pvalue))

        # Correct the nine planned feature tests within each MAT recording.
        q_values = bh_fdr(p_values)
        for row, q_value in zip(dataset_rows, q_values):
            row["BH_FDR_q"] = float(q_value)
            rows.append(row)
    return pd.DataFrame(rows)


def make_classifier(name):
    if name == "ShrinkageLDA":
        classifier = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    elif name == "LinearSVM":
        classifier = SVC(kernel="linear", C=1.0, class_weight="balanced")
    else:
        raise ValueError(f"未知分类器：{name}")
    # The scaler is fit inside each training fold, never on held-out records.
    return make_pipeline(StandardScaler(), classifier)


def classification_validation(feature_df):
    rows = []
    metric_names = ("Accuracy", "BalancedAccuracy", "AUC", "F1")

    for task, datasets in DATASETS.items():
        task_df = feature_df.loc[feature_df["Task"] == task]
        for test_dataset in datasets:
            train = task_df.loc[task_df["Dataset"] != test_dataset]
            test = task_df.loc[task_df["Dataset"] == test_dataset]
            if train.empty or test.empty:
                raise ValueError(f"{task}: 留出记录 {test_dataset} 的训练集或测试集为空")

            y_train = train["Label"].to_numpy(dtype=int)
            y_test = test["Label"].to_numpy(dtype=int)
            if not np.array_equal(np.unique(y_train), np.array([-1, 1])):
                raise ValueError(f"{task}: 训练记录必须同时含 -1/+1 标签")
            if not np.array_equal(np.unique(y_test), np.array([-1, 1])):
                raise ValueError(f"{test_dataset}: 测试记录必须同时含 -1/+1 标签")

            train_counts = {label: int(np.count_nonzero(y_train == label)) for label in (-1, 1)}
            test_counts = {label: int(np.count_nonzero(y_test == label)) for label in (-1, 1)}
            for feature_set_name, feature_names in FEATURE_SETS.items():
                x_train = train[feature_names].to_numpy(dtype=float)
                x_test = test[feature_names].to_numpy(dtype=float)
                if not np.all(np.isfinite(x_train)) or not np.all(np.isfinite(x_test)):
                    raise ValueError(f"{task}/{feature_set_name}: 分类输入含 NaN/Inf")

                for model_name in CLASSIFIERS:
                    model = make_classifier(model_name)
                    model.fit(x_train, y_train)
                    prediction = model.predict(x_test)
                    scores = model.decision_function(x_test)
                    if model[-1].classes_[1] != 1:
                        raise ValueError(f"{model_name}: 正类分数方向不是 Label=+1")
                    tn, fp, fn, tp = confusion_matrix(
                        y_test, prediction, labels=[-1, 1]
                    ).ravel()

                    row = {
                        "RowType": "fold",
                        "Task": task,
                        "TrainDataset": "+".join(d for d in datasets if d != test_dataset),
                        "TestDataset": test_dataset,
                        "TrainLeftN": train_counts[-1],
                        "TrainRightN": train_counts[1],
                        "TestLeftN": test_counts[-1],
                        "TestRightN": test_counts[1],
                        "FeatureSet": feature_set_name,
                        "Model": model_name,
                        "Accuracy": accuracy_score(y_test, prediction),
                        "BalancedAccuracy": balanced_accuracy_score(y_test, prediction),
                        "AUC": roc_auc_score(y_test, scores),
                        "F1": f1_score(y_test, prediction, pos_label=1, zero_division=0),
                        "TN": int(tn),
                        "FP": int(fp),
                        "FN": int(fn),
                        "TP": int(tp),
                    }
                    rows.append(row)

    fold_df = pd.DataFrame(rows)
    summary = (
        fold_df.groupby(["Task", "FeatureSet", "Model"], as_index=False)[list(metric_names)]
        .mean()
    )
    fold_sd = (
        fold_df.groupby(["Task", "FeatureSet", "Model"])[list(metric_names)]
        .std(ddof=1)
        .reset_index()
    )
    for metric in metric_names:
        summary[f"{metric}_SD_across_two_folds"] = fold_sd[metric]
        fold_df[f"{metric}_SD_across_two_folds"] = np.nan

    summary["RowType"] = "mean"
    summary["TrainDataset"] = ""
    summary["TestDataset"] = ""
    for count_name in ("TrainLeftN", "TrainRightN", "TestLeftN", "TestRightN"):
        summary[count_name] = np.nan
    summary[["TN", "FP", "FN", "TP"]] = np.nan

    return pd.concat([fold_df, summary], ignore_index=True, sort=False)


def plot_effect_size(stats_df):
    colors = ("#4472C4", "#ED7D31")
    fig, axes = plt.subplots(1, 2, figsize=(15, 8), sharey=True)
    y = np.arange(len(ALL_FEATURES))
    offsets = (-0.18, 0.18)

    for ax, task in zip(axes, DATASETS):
        for offset, color, dataset in zip(offsets, colors, DATASETS[task]):
            part = stats_df.loc[
                (stats_df["Task"] == task) & (stats_df["Dataset"] == dataset)
            ].set_index("Feature")
            values = part.reindex(ALL_FEATURES)["Cohens_d"].to_numpy(dtype=float)
            ax.barh(y + offset, values, height=0.34, color=color, label=dataset)
        ax.axvline(0.0, color="black", linewidth=0.8)
        ax.set_title(task)
        ax.set_xlabel("Cohen's d（左刺激 − 右刺激）")
        ax.grid(axis="x", alpha=0.25)
        ax.legend(title="记录")

    axes[0].set_yticks(y, labels=ALL_FEATURES)
    axes[0].invert_yaxis()
    fig.suptitle("左右刺激特征效应量：逐记录显示")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "特征效应量.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_classification(result_df):
    folds = result_df.loc[result_df["RowType"] == "fold"]
    metrics = ("AUC", "BalancedAccuracy")
    model_colors = {"ShrinkageLDA": "#4472C4", "LinearSVM": "#ED7D31"}
    markers = ("o", "s")
    x = np.arange(len(FEATURE_SETS))

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharey=True)
    for row_index, task in enumerate(DATASETS):
        for col_index, metric in enumerate(metrics):
            ax = axes[row_index, col_index]
            for model_name, color in model_colors.items():
                for marker, test_dataset in zip(markers, DATASETS[task]):
                    part = folds.loc[
                        (folds["Task"] == task)
                        & (folds["Model"] == model_name)
                        & (folds["TestDataset"] == test_dataset)
                    ].set_index("FeatureSet").reindex(FEATURE_SETS.keys())
                    ax.plot(
                        x,
                        part[metric].to_numpy(dtype=float),
                        color=color,
                        marker=marker,
                        linewidth=1.5,
                        label=f"{model_name} / {test_dataset}",
                    )
            ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
            ax.set_xticks(x, labels=list(FEATURE_SETS.keys()))
            ax.set_ylim(0.0, 1.0)
            ax.set_title(f"{task} · {metric}")
            ax.set_ylabel(metric)
            ax.grid(axis="y", alpha=0.25)

    # Each line is one specific held-out direction; no mean-only plot hides it.
    from matplotlib.lines import Line2D

    model_handles = [
        Line2D([], [], color=color, linestyle="-", label=model_name)
        for model_name, color in model_colors.items()
    ]
    test_handles = [
        Line2D(
            [],
            [],
            color="black",
            marker=marker,
            linestyle="None",
            label=(
                f"测试记录第{index + 1}条："
                f"{DATASETS['Task1'][index]} / {DATASETS['Task2'][index]}"
            ),
        )
        for index, marker in enumerate(markers)
    ]
    fig.legend(handles=model_handles + test_handles, loc="lower center", ncol=2, fontsize=8)
    fig.suptitle("按记录留出的分类性能（每条线对应一个测试记录方向）")
    fig.tight_layout(rect=(0, 0.09, 1, 0.95))
    fig.savefig(OUTPUT_DIR / "分类性能.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    feature_df = load_all_features()
    audit_feature_quality(feature_df)
    stats_df = feature_statistics(feature_df)
    result_df = classification_validation(feature_df)

    feature_df.to_csv(OUTPUT_DIR / "单试次特征.csv", index=False, encoding="utf-8-sig")
    stats_df.to_csv(OUTPUT_DIR / "特征统计检验.csv", index=False, encoding="utf-8-sig")
    result_df.to_csv(OUTPUT_DIR / "分类验证结果.csv", index=False, encoding="utf-8-sig")
    plot_effect_size(stats_df)
    plot_classification(result_df)

    print(f"\n06-08 特征验证完成：{OUTPUT_DIR}")
    print(f"输出 Trial 数：{len(feature_df)}；特征数：{len(ALL_FEATURES)}")
    for dataset, group in feature_df.groupby("Dataset", sort=False):
        left_n = int(np.count_nonzero(group["Label"].to_numpy() == -1))
        right_n = int(np.count_nonzero(group["Label"].to_numpy() == 1))
        print(f"{dataset}: 左={left_n}, 右={right_n}")
    print("分类结果 CSV 中 RowType=fold 为两个独立留出方向，RowType=mean 为描述性均值。")


if __name__ == "__main__":
    main()
