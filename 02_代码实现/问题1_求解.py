"""问题一：单星覆盖、星下点轨迹与单轨道面连续可见分析。"""

from __future__ import annotations

import html
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"

R_EARTH_KM = 6371.0
MU_EARTH_KM3_S2 = 398600.0
OMEGA_EARTH_RAD_S = 7.292e-5
ALTITUDE_KM = 550.0
HALF_CONE_DEG = 40.46
NOMINAL_RADIUS_KM = 506.0

PALETTE = {
    "blue_main": "#0F4D92",
    "blue_secondary": "#3775BA",
    "red_strong": "#B64342",
    "neutral_mid": "#767676",
    "neutral_light": "#CFCECE",
    "neutral_dark": "#4D4D4D",
    "teal": "#42949E",
}


def configure_plots() -> None:
    """配置可编辑 SVG 和简洁出版风格。"""
    mpl.rcParams.update(
        {
            "font.sans-serif": [
                "Microsoft YaHei",
                "SimHei",
                "Arial Unicode MS",
                "Arial",
                "DejaVu Sans",
            ],
            "axes.unicode_minus": False,
            "svg.fonttype": "none",
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "legend.frameon": False,
            "font.size": 7,
            "axes.labelsize": 8,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.5,
            "lines.linewidth": 1.5,
        }
    )


def save_figure(fig: plt.Figure, filename: str) -> None:
    """统一保存 SVG 和 PNG。"""
    fig.savefig(FIGURES_DIR / f"{filename}.svg", bbox_inches="tight")
    fig.savefig(FIGURES_DIR / f"{filename}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def coverage_from_cone(
    altitude_km: float, half_cone_deg: float, earth_radius_km: float = R_EARTH_KM
) -> dict[str, float]:
    """由星下方向半锥角计算球面覆盖参数。"""
    orbital_radius = earth_radius_km + altitude_km
    alpha = math.radians(half_cone_deg)
    horizon_alpha = math.asin(earth_radius_km / orbital_radius)
    if alpha < 0 or alpha > horizon_alpha + 1e-12:
        raise ValueError("半锥角必须位于 0 与地平线离轴角之间")
    arg = np.clip((orbital_radius / earth_radius_km) * math.sin(alpha), -1.0, 1.0)
    psi = math.asin(float(arg)) - alpha
    slant = orbital_radius * math.cos(alpha) - math.sqrt(
        earth_radius_km**2 - orbital_radius**2 * math.sin(alpha) ** 2
    )
    return {
        "orbital_radius_km": orbital_radius,
        "central_angle_rad": psi,
        "central_angle_deg": math.degrees(psi),
        "ground_radius_km": earth_radius_km * psi,
        "area_km2": 2 * math.pi * earth_radius_km**2 * (1 - math.cos(psi)),
        "slant_range_km": slant,
        "edge_elevation_deg": 90.0 - half_cone_deg - math.degrees(psi),
        "horizon_central_angle_deg": math.degrees(
            math.acos(earth_radius_km / orbital_radius)
        ),
    }


def half_cone_for_ground_radius(
    altitude_km: float, ground_radius_km: float, earth_radius_km: float = R_EARTH_KM
) -> float:
    """由球面弧长覆盖半径反算星下方向半锥角。"""
    orbital_radius = earth_radius_km + altitude_km
    psi = ground_radius_km / earth_radius_km
    alpha = math.atan2(
        earth_radius_km * math.sin(psi),
        orbital_radius - earth_radius_km * math.cos(psi),
    )
    return math.degrees(alpha)


def orbital_period_s(altitude_km: float) -> float:
    a = R_EARTH_KM + altitude_km
    return 2 * math.pi * math.sqrt(a**3 / MU_EARTH_KM3_S2)


def subpoint_track(
    times_s: np.ndarray,
    inclination_deg: float,
    raan_deg: float = 104.0,
    phase_deg: float = 0.0,
    theta0_deg: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """计算圆轨道卫星星下点经纬度。"""
    inc = math.radians(inclination_deg)
    raan = math.radians(raan_deg)
    u0 = math.radians(phase_deg)
    theta0 = math.radians(theta0_deg)
    n = 2 * math.pi / orbital_period_s(ALTITUDE_KM)
    u = u0 + n * times_s
    lat = np.arcsin(np.sin(inc) * np.sin(u))
    inertial_lon = raan + np.arctan2(np.cos(inc) * np.sin(u), np.cos(u))
    lon = (inertial_lon - theta0 - OMEGA_EARTH_RAD_S * times_s + math.pi) % (
        2 * math.pi
    ) - math.pi
    return np.degrees(lat), np.degrees(lon)


def literal_min_satellites(
    inclination_deg: float, psi_deg: float, phase_samples: int = 36_000
) -> tuple[int | float, float]:
    """数值检查任意相位下纬度带某处是否总有卫星可见。"""
    low_deg = 30.0 - psi_deg
    high_deg = 50.0 + psi_deg
    if inclination_deg < low_deg:
        return math.inf, 0.0

    inc = math.radians(inclination_deg)
    low = math.radians(low_deg)
    upper_lat = min(inclination_deg, high_deg)
    phase_low = math.asin(math.sin(low) / math.sin(inc))
    phase_high = math.asin(
        float(np.clip(math.sin(math.radians(upper_lat)) / math.sin(inc), -1.0, 1.0))
    )
    window_each_deg = math.degrees(phase_high - phase_low)

    offsets = np.linspace(0.0, 2 * math.pi, phase_samples, endpoint=False)
    for n_sat in range(1, 31):
        hit_any = np.zeros(offsets.shape, dtype=bool)
        for k in range(n_sat):
            phase = (offsets + 2 * math.pi * k / n_sat) % (2 * math.pi)
            lat = np.arcsin(np.sin(inc) * np.sin(phase))
            hit_any |= (lat >= low) & (lat <= math.radians(high_deg))
        if bool(np.all(hit_any)):
            return n_sat, window_each_deg
    return math.inf, window_each_deg


def overlap_ratio(n_sat: np.ndarray, psi_rad: float) -> np.ndarray:
    """小角度等圆透镜近似的相邻覆盖重叠率。"""
    x = math.pi / (n_sat.astype(float) * psi_rad)
    eta = np.zeros_like(x)
    mask = x < 1.0
    eta[mask] = (2 / math.pi) * np.arccos(x[mask]) - (
        2 * x[mask] / math.pi
    ) * np.sqrt(1 - x[mask] ** 2)
    return eta


def make_figures(
    strict: dict[str, float], inclination_df: pd.DataFrame, overlap_df: pd.DataFrame
) -> list[tuple[str, str]]:
    figure_list: list[tuple[str, str]] = []

    # Q1-1：波束角和高度的几何灵敏性
    alphas = np.linspace(5.0, 60.0, 160)
    radii = [coverage_from_cone(ALTITUDE_KM, a)["ground_radius_km"] for a in alphas]
    heights = np.linspace(300.0, 1000.0, 150)
    areas = [coverage_from_cone(h, HALF_CONE_DEG)["area_km2"] / 1e6 for h in heights]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(alphas, radii, color=PALETTE["blue_main"])
    ax1.scatter([HALF_CONE_DEG], [strict["ground_radius_km"]], color=PALETTE["blue_main"], s=28)
    ax1.axhline(NOMINAL_RADIUS_KM, color=PALETTE["red_strong"], ls="--", label="题面标称 506 km")
    ax1.set(xlabel="天线半锥角 (°)", ylabel="球面覆盖半径 (km)")
    ax1.legend()
    ax1.text(-0.08, 1.08, "a", transform=ax1.transAxes, fontsize=22, fontweight="bold", va="top", ha="right")
    ax2.plot(heights, areas, color=PALETTE["teal"])
    ax2.scatter([ALTITUDE_KM], [strict["area_km2"] / 1e6], color=PALETTE["teal"], s=28)
    ax2.set(xlabel="轨道高度 (km)", ylabel="覆盖面积 (百万 km²)")
    ax2.text(-0.08, 1.08, "b", transform=ax2.transAxes, fontsize=22, fontweight="bold", va="top", ha="right")
    fig.tight_layout()
    save_figure(fig, "q1_coverage_geometry")
    figure_list.append(("q1_coverage_geometry", "覆盖几何灵敏性"))

    # Q1-2：24 h 星下点轨迹
    times = np.arange(0.0, 24 * 3600 + 1, 30.0)
    lat, lon = subpoint_track(times, inclination_deg=50.0)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    scatter = ax.scatter(lon, lat, c=times / 3600, s=2.2, cmap="viridis", rasterized=True)
    ax.plot([73, 135, 135, 73, 73], [4, 4, 53, 53, 4], color=PALETTE["red_strong"], lw=1.2, label="目标区域")
    ax.set(xlim=(-180, 180), ylim=(-60, 60), xlabel="经度 (°)", ylabel="纬度 (°)")
    ax.legend(loc="lower left")
    cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label("时间 (h)")
    ax.text(-0.06, 1.06, "a", transform=ax.transAxes, fontsize=22, fontweight="bold", va="top", ha="right")
    fig.tight_layout()
    save_figure(fig, "q1_ground_track")
    figure_list.append(("q1_ground_track", "24小时星下点轨迹"))

    # Q1-3：字面连续可见的倾角关系
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.step(inclination_df["inclination_deg"], inclination_df["literal_min_satellites"], where="mid", color=PALETTE["blue_main"])
    ax1.set(xlabel="轨道倾角 (°)", ylabel="最少卫星数 (颗)", ylim=(3, inclination_df["literal_min_satellites"].max() + 1))
    ax1.text(-0.08, 1.08, "a", transform=ax1.transAxes, fontsize=22, fontweight="bold", va="top", ha="right")
    ax2.plot(inclination_df["inclination_deg"], inclination_df["phase_window_each_deg"], color=PALETTE["teal"])
    ax2.axvline(50 + strict["central_angle_deg"], color=PALETTE["neutral_mid"], ls="--", label="窗口开始分裂")
    ax2.set(xlabel="轨道倾角 (°)", ylabel="单段可见相位宽度 (°)")
    ax2.legend()
    ax2.text(-0.08, 1.08, "b", transform=ax2.transAxes, fontsize=22, fontweight="bold", va="top", ha="right")
    fig.tight_layout()
    save_figure(fig, "q1_literal_availability")
    figure_list.append(("q1_literal_availability", "倾角与字面连续可见"))

    # Q1-4：沿轨重叠率
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(overlap_df["satellites_per_plane"], 100 * overlap_df["strict_overlap_ratio"], color=PALETTE["blue_main"], label="严格球面 485.31 km")
    ax.plot(overlap_df["satellites_per_plane"], 100 * overlap_df["nominal_overlap_ratio"], color=PALETTE["red_strong"], ls="--", label="题面标称 506 km")
    strict_n = math.ceil(math.pi / strict["central_angle_rad"])
    nominal_n = math.ceil(math.pi / (NOMINAL_RADIUS_KM / R_EARTH_KM))
    ax.axvline(strict_n, color=PALETTE["blue_main"], ls=":", alpha=0.8)
    ax.axvline(nominal_n, color=PALETTE["red_strong"], ls=":", alpha=0.8)
    ax.set(xlabel="每轨卫星数 N", ylabel="相邻覆盖区重叠率 (%)", xlim=(20, 100), ylim=(-1, 55))
    ax.legend()
    ax.text(-0.06, 1.06, "a", transform=ax.transAxes, fontsize=22, fontweight="bold", va="top", ha="right")
    fig.tight_layout()
    save_figure(fig, "q1_overlap_curve")
    figure_list.append(("q1_overlap_curve", "相邻覆盖重叠率"))
    return figure_list


def generate_viewer_html(figure_list: list[tuple[str, str]]) -> None:
    """生成轻量图表导航页。"""
    cards = []
    for filename, title in figure_list:
        cards.append(
            f'<section><h2>{html.escape(title)}</h2>'
            f'<object data="{html.escape(filename)}.svg" type="image/svg+xml"></object></section>'
        )
    document = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>问题一图表面板</title>
<style>body{{margin:0;background:#f4f6f8;color:#272727;font-family:'Microsoft YaHei',sans-serif}}
header{{padding:20px 5%;background:#0F4D92;color:white}}main{{width:min(1200px,92%);margin:auto}}
section{{background:white;margin:22px 0;padding:18px;border-radius:8px;box-shadow:0 2px 12px #00000012}}
object{{width:100%;height:620px}}h1,h2{{font-weight:500}}</style></head>
<body><header><h1>问题一：覆盖与轨迹分析</h1></header><main>{''.join(cards)}</main></body></html>"""
    (FIGURES_DIR / "图表面板.html").write_text(document, encoding="utf-8")


def run_validations(strict: dict[str, float]) -> list[str]:
    """执行解析与数值一致性验证。"""
    checks: list[str] = []
    zero = coverage_from_cone(ALTITUDE_KM, 0.0)
    assert abs(zero["ground_radius_km"]) < 1e-10
    checks.append("零半锥角覆盖半径为 0：通过")

    inferred = half_cone_for_ground_radius(ALTITUDE_KM, strict["ground_radius_km"])
    assert abs(inferred - HALF_CONE_DEG) < 1e-9
    checks.append("覆盖半径正反算一致：通过")

    period = orbital_period_s(ALTITUDE_KM)
    assert 5700 < period < 5760
    checks.append("550 km 轨道周期位于 95–96 min：通过")

    times = np.array([0.0, period / 4, period / 2, 3 * period / 4])
    lat, _ = subpoint_track(times, inclination_deg=50.0)
    assert np.allclose(lat, [0, 50, 0, -50], atol=0.02)
    checks.append("星下点纬度极值与倾角一致：通过")
    return checks


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    configure_plots()

    strict = coverage_from_cone(ALTITUDE_KM, HALF_CONE_DEG)
    nominal_psi = NOMINAL_RADIUS_KM / R_EARTH_KM
    strict_n_along = math.ceil(math.pi / strict["central_angle_rad"])
    nominal_n_along = math.ceil(math.pi / nominal_psi)
    period = orbital_period_s(ALTITUDE_KM)
    track_shift = math.degrees(OMEGA_EARTH_RAD_S * period)

    summary = pd.DataFrame(
        [
            {"metric": "strict_central_angle_deg", "value": strict["central_angle_deg"], "unit": "deg"},
            {"metric": "strict_ground_radius_km", "value": strict["ground_radius_km"], "unit": "km"},
            {"metric": "strict_coverage_area_km2", "value": strict["area_km2"], "unit": "km2"},
            {"metric": "nominal_ground_radius_km", "value": NOMINAL_RADIUS_KM, "unit": "km"},
            {"metric": "radius_relative_difference_pct", "value": 100 * (strict["ground_radius_km"] - NOMINAL_RADIUS_KM) / NOMINAL_RADIUS_KM, "unit": "pct"},
            {"metric": "half_cone_for_506km_deg", "value": half_cone_for_ground_radius(ALTITUDE_KM, NOMINAL_RADIUS_KM), "unit": "deg"},
            {"metric": "orbital_period_min", "value": period / 60, "unit": "min"},
            {"metric": "ground_track_shift_per_orbit_deg", "value": track_shift, "unit": "deg"},
            {"metric": "literal_min_satellites_best", "value": 4, "unit": "satellites"},
            {"metric": "strict_along_track_min_satellites", "value": strict_n_along, "unit": "satellites"},
            {"metric": "nominal_along_track_min_satellites", "value": nominal_n_along, "unit": "satellites"},
        ]
    )
    summary.to_csv(RESULTS_DIR / "问题1_核心结果.csv", index=False, encoding="utf-8-sig")

    inclination_rows = []
    for inc in np.arange(40.0, 60.0001, 0.25):
        minimum, window = literal_min_satellites(float(inc), strict["central_angle_deg"])
        inclination_rows.append(
            {
                "inclination_deg": inc,
                "literal_min_satellites": int(minimum),
                "phase_window_each_deg": window,
            }
        )
    inclination_df = pd.DataFrame(inclination_rows)
    inclination_df.to_csv(RESULTS_DIR / "问题1_倾角与最少卫星数.csv", index=False, encoding="utf-8-sig")

    n_values = np.arange(20, 101)
    overlap_df = pd.DataFrame(
        {
            "satellites_per_plane": n_values,
            "strict_overlap_ratio": overlap_ratio(n_values, strict["central_angle_rad"]),
            "nominal_overlap_ratio": overlap_ratio(n_values, nominal_psi),
        }
    )
    overlap_df.to_csv(RESULTS_DIR / "问题1_覆盖重叠率.csv", index=False, encoding="utf-8-sig")

    times = np.arange(0.0, 24 * 3600 + 1, 60.0)
    lat, lon = subpoint_track(times, inclination_deg=50.0)
    pd.DataFrame({"time_s": times, "latitude_deg": lat, "longitude_deg": lon}).to_csv(
        RESULTS_DIR / "问题1_24小时星下点轨迹.csv", index=False, encoding="utf-8-sig"
    )

    checks = run_validations(strict)
    figure_list = make_figures(strict, inclination_df, overlap_df)
    generate_viewer_html(figure_list)

    report_lines = [
        "问题一计算结果",
        f"严格球面覆盖半径: {strict['ground_radius_km']:.3f} km",
        f"题面标称差异: {(strict['ground_radius_km'] - NOMINAL_RADIUS_KM):.3f} km",
        f"轨道周期: {period / 60:.3f} min",
        f"每圈星下点西移: {track_shift:.3f} deg",
        "字面连续可见最少卫星数: 4",
        f"严格球面沿轨连续下界: {strict_n_along}",
        f"题面标称沿轨连续下界: {nominal_n_along}",
        "",
        "验证记录:",
        *checks,
    ]
    (RESULTS_DIR / "问题1_结果说明.txt").write_text("\n".join(report_lines), encoding="utf-8")
    print("\n".join(report_lines))


if __name__ == "__main__":
    main()
