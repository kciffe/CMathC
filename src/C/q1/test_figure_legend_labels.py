import importlib.util
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt


BASE_DIR = Path(__file__).parent


def load_module(filename, module_name):
    module_path = BASE_DIR / filename
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RIEMANN = load_module("8riemann_denoise.py", "riemann_denoise_legend_test")
BOUNDARY = load_module("9check_riemann_drop.py", "riemann_boundary_legend_test")
ALPHA = load_module("10check_b_alpha_sensitivity.py", "alpha_sensitivity_legend_test")
CLEANING = load_module("12check_a1_cleaning_sensitivity.py", "a1_sensitivity_legend_test")


def test_riemann_status_legend_labels_are_chinese():
    assert RIEMANN.status_legend_labels() == ("保留试次", "黎曼阈值剔除试次")


def test_boundary_trial_legend_label_is_chinese():
    assert BOUNDARY.format_trial_sqi_legend(7, 0.421) == (
        "第 7 个试次 · 信号质量指数 0.421"
    )


def test_alpha_legend_label_describes_retained_trial_count_in_chinese():
    assert ALPHA.format_alpha_legend(0.6, 83) == "α=0.6（保留 83 个试次）"


def test_cleaning_sensitivity_legend_labels_are_chinese():
    assert CLEANING.cleaning_legend_labels(42, 29) == (
        "仅按削顶规则筛选（42 个试次）",
        "随机等样本量平均波形（29 个试次）",
        "随机等样本量抽样的逐点 95% 区间",
        "当前质量指数筛选结果（29 个试次）",
    )


def assert_legend_is_below_title_and_above_axes(module, labels):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("图标题\n图注说明", fontsize=13)
    source_axis = axes[0, 0]
    for index, label in enumerate(labels):
        source_axis.plot([0, 1], [index, index + 1], label=label)

    helper = getattr(module, "install_external_legend", None)
    assert callable(helper), "the plotter should expose its external-legend layout"
    legend = helper(fig, source_axis, fontsize=8)
    fig.canvas.draw()

    axes_boxes = [axis.get_window_extent() for axis in axes.flat]
    axes_top = max(box.y1 for box in axes_boxes)
    axes_right = max(box.x1 for box in axes_boxes)
    legend_box = legend.get_window_extent()
    title_box = fig._suptitle.get_window_extent()
    figure_box = fig.bbox
    plt.close(fig)
    assert axes_top <= figure_box.y1 * 0.88
    assert axes_right >= figure_box.x1 * 0.95
    assert legend_box.y0 >= axes_top
    assert legend_box.y1 <= title_box.y0
    assert legend_box.y0 - axes_top <= figure_box.y1 * 0.10


def test_alpha_sensitivity_legend_is_outside_the_plot_grid():
    labels = [ALPHA.format_alpha_legend(value, 80) for value in (0.6, 0.7, 0.8, 1.0)]
    assert_legend_is_below_title_and_above_axes(ALPHA, labels)


def test_cleaning_sensitivity_legend_is_outside_the_plot_grid():
    labels = CLEANING.cleaning_legend_labels(42, 29)
    assert_legend_is_below_title_and_above_axes(CLEANING, labels)
