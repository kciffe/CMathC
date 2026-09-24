# -*- coding: utf-8 -*-
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_DIR = SCRIPT_DIR / "output" / "06_08_feature_validation"
OUTPUT_DIR = SCRIPT_DIR / "output" / "09_feature_diagnosis"

FEATURE_FILE = INPUT_DIR / "单试次特征.csv"
CROSS_RECORD_FILE = INPUT_DIR / "分类验证结果.csv"

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
    "ERP+空间+频谱": FEATURES_ERP + FEATURES_SPATIAL + FEATURES_SPECTRAL,
}

# 置换检验预先固定，不根据当前结果挑模型
PRIMARY_FEATURE_SET = "ERP+空间"
PRIMARY_MODEL = "ShrinkageLDA"

N_SPLITS = 5
N_PERMUTATIONS = 1000
RANDOM_STATE = 2026
TASK_DATASETS = {
    "Task1": ["VisualCogA_Task-1", "VisualCogA_Task-2"],
    "Task2": ["VisualCogB_Task-1", "VisualCogB_Task-2"],
}

plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False


def make_classifier(name):
    if name == "ShrinkageLDA":
        return make_pipeline(
            StandardScaler(),
            LinearDiscriminantAnalysis(
                solver="lsqr",
                shrinkage="auto",
            ),
        )

    if name == "LinearSVM":
        return make_pipeline(
            StandardScaler(),
            SVC(
                kernel="linear",
                C=1.0,
                class_weight="balanced",
            ),
        )
    raise ValueError(f"未知分类器：{name}")


def evaluate_fold(model, x_train, y_train, x_test, y_test):
    model.fit(x_train, y_train)

    pred = model.predict(x_test)
    score = model.decision_function(x_test)
    if model[-1].classes_[1] != 1:
        raise ValueError("分类器 decision_function 的正方向不是 Label=+1")

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        pred,
        labels=[-1, 1],
    ).ravel()

    return {
        "Accuracy": accuracy_score(y_test, pred),
        "BalancedAccuracy": balanced_accuracy_score(y_test, pred),
        "AUC": roc_auc_score(y_test, score),
        "F1": f1_score(y_test, pred, pos_label=1, zero_division=0),
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
    }


def within_record_validation(feature_df):
    rows = []

    for dataset, group in feature_df.groupby("Dataset", sort=False):
        task = group["Task"].iloc[0]
        y = group["Label"].to_numpy()

        splitter = StratifiedKFold(
            n_splits=N_SPLITS,
            shuffle=True,
            random_state=RANDOM_STATE,
        )
        splits = list(splitter.split(np.zeros(len(y)), y))

        for feature_set, feature_names in FEATURE_SETS.items():
            x = group[feature_names].to_numpy(dtype=float)
            if not np.all(np.isfinite(x)):
                raise ValueError(f"{dataset}/{feature_set}: 特征含 NaN/Inf")

            for model_name in ["ShrinkageLDA", "LinearSVM"]:
                fold_rows = []

                for fold, (train_idx, test_idx) in enumerate(splits, start=1):
                    result = evaluate_fold(
                        make_classifier(model_name),
                        x[train_idx],
                        y[train_idx],
                        x[test_idx],
                        y[test_idx],
                    )

                    row = {
                        "RowType": "fold",
                        "Task": task,
                        "Dataset": dataset,
                        "Fold": fold,
                        "FeatureSet": feature_set,
                        "Model": model_name,
                        "TrainN": len(train_idx),
                        "TestN": len(test_idx),
                        "TrainLeftN": int(np.sum(y[train_idx] == -1)),
                        "TrainRightN": int(np.sum(y[train_idx] == 1)),
                        "TestLeftN": int(np.sum(y[test_idx] == -1)),
                        "TestRightN": int(np.sum(y[test_idx] == 1)),
                        **result,
                    }

                    rows.append(row)
                    fold_rows.append(row)

                summary = {
                    "RowType": "mean",
                    "Task": task,
                    "Dataset": dataset,
                    "Fold": "",
                    "FeatureSet": feature_set,
                    "Model": model_name,
                    "TrainN": "",
                    "TestN": "",
                    "TestLeftN": "",
                    "TestRightN": "",
                }

                for metric in [
                    "Accuracy",
                    "BalancedAccuracy",
                    "AUC",
                    "F1",
                ]:
                    values = np.array(
                        [row[metric] for row in fold_rows],
                        dtype=float,
                    )
                    summary[metric] = values.mean()
                    summary[f"{metric}_SD"] = values.std(ddof=1)

                for key in ["TN", "FP", "FN", "TP"]:
                    summary[key] = int(
                        np.sum([row[key] for row in fold_rows])
                    )

                rows.append(summary)

    return pd.DataFrame(rows)


def mean_cv_auc(x, y, splits, model_name):
    aucs = []

    for train_idx, test_idx in splits:
        model = make_classifier(model_name)
        model.fit(x[train_idx], y[train_idx])
        if model[-1].classes_[1] != 1:
            raise ValueError("分类器 decision_function 的正方向不是 Label=+1")
        score = model.decision_function(x[test_idx])
        aucs.append(
            roc_auc_score(
                y[test_idx],
                score,
            )
        )

    return float(np.mean(aucs))


def permutation_test(feature_df):
    rows = []
    rng = np.random.default_rng(RANDOM_STATE)
    feature_names = FEATURE_SETS[PRIMARY_FEATURE_SET]

    for dataset, group in feature_df.groupby("Dataset", sort=False):
        task = group["Task"].iloc[0]
        x = group[feature_names].to_numpy(dtype=float)
        y = group["Label"].to_numpy(dtype=int)
        if not np.all(np.isfinite(x)):
            raise ValueError(f"{dataset}/{PRIMARY_FEATURE_SET}: 置换检验输入含 NaN/Inf")

        splitter = StratifiedKFold(
            n_splits=N_SPLITS,
            shuffle=True,
            random_state=RANDOM_STATE,
        )
        observed_splits = list(splitter.split(np.zeros(len(y)), y))

        observed_auc = mean_cv_auc(
            x,
            y,
            observed_splits,
            PRIMARY_MODEL,
        )

        perm_auc = np.empty(N_PERMUTATIONS, dtype=float)

        for i in range(N_PERMUTATIONS):
            y_perm = rng.permutation(y)
            # Rebuild stratified folds from each permuted label vector. This
            # repeats the complete CV procedure under the null and guarantees
            # that every fold contains both classes; it avoids reusing folds
            # whose membership was stratified on the observed labels.
            perm_splits = list(splitter.split(np.zeros(len(y_perm)), y_perm))
            perm_auc[i] = mean_cv_auc(
                x,
                y_perm,
                perm_splits,
                PRIMARY_MODEL,
            )

        p_value = (
            1
            + np.sum(perm_auc >= observed_auc)
        ) / (
            N_PERMUTATIONS + 1
        )

        rows.append({
            "Task": task,
            "Dataset": dataset,
            "FeatureSet": PRIMARY_FEATURE_SET,
            "Model": PRIMARY_MODEL,
            "ObservedMeanAUC": observed_auc,
            "PermutationMeanAUC": float(perm_auc.mean()),
            "PermutationAUC_2.5%": float(np.percentile(perm_auc, 2.5)),
            "PermutationAUC_97.5%": float(np.percentile(perm_auc, 97.5)),
            "PermutationP": float(p_value),
            "NullAUC_95thPercentile": float(np.percentile(perm_auc, 95)),
            "NullExceedances": int(np.sum(perm_auc >= observed_auc)),
            "NPermutations": N_PERMUTATIONS,
            "RandomState": RANDOM_STATE,
            "Alternative": "one-sided: observed AUC > permuted AUC",
        })

    return pd.DataFrame(rows)


def validate_inputs(feature_df, cross_df):
    required_feature_columns = {
        "Task",
        "Dataset",
        "Trial",
        "Label",
        *[feature for feature_names in FEATURE_SETS.values() for feature in feature_names],
    }
    missing = sorted(required_feature_columns.difference(feature_df.columns))
    if missing:
        raise ValueError(f"单试次特征 CSV 缺少字段：{missing}")

    expected = {
        dataset: task
        for task, datasets in TASK_DATASETS.items()
        for dataset in datasets
    }
    found = set(feature_df["Dataset"].dropna().astype(str).unique())
    if found != set(expected):
        raise ValueError(f"数据集应为 {sorted(expected)}，实际为 {sorted(found)}")
    if not np.all(np.isin(feature_df["Label"].to_numpy(), (-1, 1))):
        raise ValueError("Label 只能是 -1/+1")

    for dataset, group in feature_df.groupby("Dataset", sort=False):
        expected_task = expected[dataset]
        if set(group["Task"].astype(str).unique()) != {expected_task}:
            raise ValueError(f"{dataset}: Task 与数据集名称不一致")
        counts = group["Label"].value_counts()
        if min(int(counts.get(-1, 0)), int(counts.get(1, 0))) < N_SPLITS:
            raise ValueError(f"{dataset}: 左右任一类别少于 {N_SPLITS} 个 Trial")
        if group["Trial"].duplicated().any():
            raise ValueError(f"{dataset}: Trial 编号重复")
        values = group[list(ALL_FEATURES)].to_numpy(dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{dataset}: 九个特征含 NaN/Inf")

    required_cross_columns = {
        "RowType",
        "Task",
        "TestDataset",
        "FeatureSet",
        "Model",
        "AUC",
        "BalancedAccuracy",
    }
    missing_cross = sorted(required_cross_columns.difference(cross_df.columns))
    if missing_cross:
        raise ValueError(f"跨记录分类结果 CSV 缺少字段：{missing_cross}")
    primary_cross = cross_df.loc[
        (cross_df["RowType"] == "fold")
        & (cross_df["FeatureSet"] == PRIMARY_FEATURE_SET)
        & (cross_df["Model"] == PRIMARY_MODEL)
    ]
    if set(primary_cross["TestDataset"].astype(str)) != set(expected):
        raise ValueError("跨记录结果没有覆盖四个测试记录的主诊断模型结果")
    if primary_cross["TestDataset"].duplicated().any():
        raise ValueError("跨记录结果中主诊断模型的测试记录重复")
    if not np.all(
        np.isfinite(primary_cross[["AUC", "BalancedAccuracy"]].to_numpy(dtype=float))
    ):
        raise ValueError("跨记录主诊断模型结果含 NaN/Inf")


def plot_within_vs_cross(within_df, cross_df):
    within = within_df[
        (within_df["RowType"] == "mean")
        & (within_df["FeatureSet"] == PRIMARY_FEATURE_SET)
        & (within_df["Model"] == PRIMARY_MODEL)
    ][
        [
            "Task",
            "Dataset",
            "AUC",
            "BalancedAccuracy",
            "AUC_SD",
            "BalancedAccuracy_SD",
        ]
    ].copy()

    cross = cross_df[
        (cross_df["RowType"] == "fold")
        & (cross_df["FeatureSet"] == PRIMARY_FEATURE_SET)
        & (cross_df["Model"] == PRIMARY_MODEL)
    ][
        [
            "Task",
            "TestDataset",
            "AUC",
            "BalancedAccuracy",
        ]
    ].copy()

    cross = cross.rename(
        columns={"TestDataset": "Dataset"}
    )

    merged = within.merge(
        cross,
        on=["Task", "Dataset"],
        suffixes=("_within", "_cross"),
    )

    merged["显示名"] = merged["Dataset"].str.replace(
        "VisualCog",
        "",
        regex=False,
    )

    x = np.arange(len(merged))
    width = 0.36

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(12, 9),
        sharex=True,
    )

    for ax, metric, title in [
        (axes[0], "AUC", "AUC：记录内 5 折 vs 跨记录留出"),
        (
            axes[1],
            "BalancedAccuracy",
            "Balanced Accuracy：记录内 5 折 vs 跨记录留出",
        ),
    ]:
        ax.bar(
            x - width / 2,
            merged[f"{metric}_within"],
            width,
            label="记录内5折",
            yerr=merged[f"{metric}_SD"],
            capsize=3,
        )
        ax.bar(
            x + width / 2,
            merged[f"{metric}_cross"],
            width,
            label="跨记录留出",
        )
        ax.axhline(
            0.5,
            linestyle="--",
            linewidth=1,
        )
        ax.set_ylim(0, 1)
        ax.set_ylabel(metric)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        ax.legend()

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(
        merged["显示名"],
        rotation=15,
    )

    fig.suptitle(
        f"{PRIMARY_MODEL} + {PRIMARY_FEATURE_SET}：记录内信号与跨记录泛化对比"
    )
    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR / "记录内与跨记录对比.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not FEATURE_FILE.is_file():
        raise FileNotFoundError(f"找不到单试次特征文件：{FEATURE_FILE}")
    if not CROSS_RECORD_FILE.is_file():
        raise FileNotFoundError(f"找不到跨记录分类文件：{CROSS_RECORD_FILE}")
    feature_df = pd.read_csv(FEATURE_FILE, encoding="utf-8-sig")
    cross_df = pd.read_csv(CROSS_RECORD_FILE, encoding="utf-8-sig")
    validate_inputs(feature_df, cross_df)

    within_df = within_record_validation(feature_df)
    within_df.to_csv(
        OUTPUT_DIR / "记录内分类诊断.csv",
        index=False,
        encoding="utf-8-sig",
    )

    permutation_df = permutation_test(feature_df)
    permutation_df.to_csv(
        OUTPUT_DIR / "置换检验.csv",
        index=False,
        encoding="utf-8-sig",
    )

    plot_within_vs_cross(
        within_df,
        cross_df,
    )

    print("09 特征诊断完成：", OUTPUT_DIR)
    print()
    print("主要诊断模型：", PRIMARY_MODEL, "+", PRIMARY_FEATURE_SET)
    print()
    print(
        permutation_df[
            [
                "Dataset",
                "ObservedMeanAUC",
                "PermutationP",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
