# -*- coding: utf-8 -*-
"""
功能：
1. 读取指定目录下所有 shp 文件
2. 统一坐标系并保存到一个 GeoPackage(.gpkg)
3. 将行政区、水系等图层组合绘制成一张地图图片

输出：
1. all_gis_layers.gpkg
2. 地理信息组合图.png
"""

from pathlib import Path
import geopandas as gpd
import matplotlib.pyplot as plt


# =========================
# 1. 路径设置
# =========================
DATA_ROOT = Path(r"D:\8\Desktop\CMathc\data\D题\题目数据及检验\随题数据\地理信息数据")
OUTPUT_DIR = Path(r"D:\8\Desktop\CMathc\src\utils\output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

GPKG_PATH = DATA_ROOT / "all_gis_layers.gpkg"
FIG_PATH = OUTPUT_DIR / "地理信息组合图.png"

TARGET_CRS = "EPSG:4326"   # 统一成经纬度


# =========================
# 2. Matplotlib 中文显示
# =========================
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


# =========================
# 3. 工具函数
# =========================
def find_name_column(gdf):
    """
    尝试寻找“名称列”，用于筛选江苏。
    常见字段可能叫 NAME / name / NAME_CHN / 省 / 市 / 县名 等
    """
    candidates = [
        "NAME", "Name", "name",
        "NAME_CHN", "name_ch", "name_chn",
        "省", "市", "县", "区",
        "省名", "市名", "县名", "名称", "中文名"
    ]
    for col in candidates:
        if col in gdf.columns:
            return col

    # 如果没有明显名称列，就从 object 类型字段里挑一个
    for col in gdf.columns:
        if gdf[col].dtype == "object":
            return col

    return None


def to_target_crs(gdf, target_crs="EPSG:4326"):
    """统一坐标系"""
    if gdf.crs is None:
        # 没有坐标系信息时，先直接返回；很多情况下仍可绘制
        return gdf
    if str(gdf.crs) != target_crs:
        return gdf.to_crs(target_crs)
    return gdf


def choose_boundary_style(layer_name):
    """
    给不同图层简单配色
    """
    if "省" in layer_name:
        return dict(facecolor="none", edgecolor="black", linewidth=1.2, alpha=0.9, zorder=2)
    elif "市" in layer_name:
        return dict(facecolor="none", edgecolor="gray", linewidth=0.7, alpha=0.8, zorder=3)
    elif "县" in layer_name:
        return dict(facecolor="none", edgecolor="lightgray", linewidth=0.4, alpha=0.7, zorder=1)
    elif "九段线" in layer_name:
        return dict(color="black", linewidth=0.8, alpha=0.8, zorder=4)
    elif "河" in layer_name:
        return dict(facecolor="#7ec8e3", edgecolor="#4a90e2", linewidth=0.5, alpha=0.8, zorder=5)
    elif "湖" in layer_name or "库" in layer_name:
        return dict(facecolor="#a8d8ff", edgecolor="#4a90e2", linewidth=0.5, alpha=0.8, zorder=5)
    else:
        return dict(facecolor="none", edgecolor="black", linewidth=0.6, alpha=0.8, zorder=2)


# =========================
# 4. 读取所有 shp，并写入 gpkg
# =========================
shp_files = list(DATA_ROOT.rglob("*.shp"))

if not shp_files:
    raise FileNotFoundError(f"在目录中没有找到 shp 文件：{DATA_ROOT}")

print(f"共找到 {len(shp_files)} 个 shp 文件：")
for shp in shp_files:
    print(" -", shp.name)

# 如果已有旧 gpkg，先删掉，避免同名图层反复叠加报错
if GPKG_PATH.exists():
    GPKG_PATH.unlink()

layers = {}  # layer_name -> GeoDataFrame

for shp in shp_files:
    layer_name = shp.stem
    gdf = gpd.read_file(shp)
    gdf = to_target_crs(gdf, TARGET_CRS)
    layers[layer_name] = gdf

    # 写入 GeoPackage
    gdf.to_file(GPKG_PATH, layer=layer_name, driver="GPKG")

print(f"\n已组合保存到：{GPKG_PATH}")


# =========================
# 5. 尝试提取“江苏省”范围
# =========================
jiangsu_geom = None

# 优先从“省”图层里找江苏
province_layer = None
for name in layers:
    if "省" in name:
        province_layer = layers[name]
        break

if province_layer is not None:
    name_col = find_name_column(province_layer)
    if name_col is not None:
        js = province_layer[province_layer[name_col].astype(str).str.contains("江苏", na=False)]
        if len(js) > 0:
            jiangsu_geom = js.unary_union
            print("\n已识别出江苏省范围。")
        else:
            print("\n未在省级图层中识别出“江苏”，将直接绘制全部范围。")
    else:
        print("\n省级图层未找到名称字段，将直接绘制全部范围。")
else:
    print("\n未找到省级图层，将直接绘制全部范围。")


# =========================
# 6. 绘图
# =========================
fig, ax = plt.subplots(figsize=(12, 10))

for layer_name, gdf in layers.items():
    plot_gdf = gdf.copy()

    # 对市/县/河流/湖泊做江苏范围裁剪（如果识别出了江苏）
    if jiangsu_geom is not None:
        if ("市" in layer_name) or ("县" in layer_name) or ("江苏" in layer_name):
            try:
                plot_gdf = gpd.clip(plot_gdf, jiangsu_geom)
            except Exception:
                pass

        # 省级图层只画江苏
        if "省" in layer_name:
            name_col = find_name_column(plot_gdf)
            if name_col is not None:
                tmp = plot_gdf[plot_gdf[name_col].astype(str).str.contains("江苏", na=False)]
                if len(tmp) > 0:
                    plot_gdf = tmp

    if len(plot_gdf) == 0:
        continue

    style = choose_boundary_style(layer_name)

    # 判断几何类型，线和面分别画
    geom_types = plot_gdf.geometry.geom_type.unique().tolist()

    if any(t in ["LineString", "MultiLineString"] for t in geom_types):
        plot_gdf.plot(ax=ax, **style)
    else:
        plot_gdf.plot(ax=ax, **style)

# 设置显示范围
if jiangsu_geom is not None:
    minx, miny, maxx, maxy = gpd.GeoSeries([jiangsu_geom], crs=TARGET_CRS).total_bounds
    dx = (maxx - minx) * 0.08
    dy = (maxy - miny) * 0.08
    ax.set_xlim(minx - dx, maxx + dx)
    ax.set_ylim(miny - dy, maxy + dy)

ax.set_title("地理信息组合图（江苏区域）", fontsize=16)
ax.set_xlabel("经度")
ax.set_ylabel("纬度")
ax.grid(True, linestyle="--", alpha=0.3)

plt.tight_layout()
plt.savefig(FIG_PATH, dpi=300, bbox_inches="tight")
plt.show()

print(f"\n地图图片已保存到：{FIG_PATH}")