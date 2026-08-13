"""问题二：多轨道面区域覆盖评估与分层确定性搜索。"""

from __future__ import annotations

import html
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"

R_EARTH_KM = 6371.0
ALTITUDE_KM = 550.0
ORBIT_RADIUS_KM = R_EARTH_KM + ALTITUDE_KM
MU_EARTH_KM3_S2 = 398600.0
OMEGA_EARTH_RAD_S = 7.292e-5
SIDEREAL_DAY_S = 86164.0905
PSI_STRICT_RAD = math.radians(4.36446872665254)
PSI_NOMINAL_RAD = 506.0 / R_EARTH_KM

LAT_RANGE = (4.0, 53.0)
LON_RANGE = (73.0, 135.0)

PALETTE = {
    "blue_main": "#0F4D92",
    "blue_secondary": "#3775BA",
    "red_strong": "#B64342",
    "neutral_mid": "#767676",
    "neutral_light": "#CFCECE",
    "teal": "#42949E",
    "green": "#8BCF8B",
}


@dataclass(frozen=True)
class Constellation:
    planes: int
    sats_per_plane: int
    inclination_deg: float
    phase_factor: int
    raan0_deg: float = 0.0
    phase0_deg: float = 0.0

    @property
    def total_satellites(self) -> int:
        return self.planes * self.sats_per_plane


@dataclass
class CoverageMetrics:
    min_multiplicity: int
    spacetime_single_ratio: float
    min_point_single_availability: float
    spacetime_double_ratio: float
    min_point_double_availability: float
    max_gap_s: float
    worst_time_index: int
    worst_time_single_ratio: float
    average_multiplicity: float
    min_margin_deg: float


def configure_plots() -> None:
    mpl.rcParams.update(
        {
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "svg.fonttype": "none",
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
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
    fig.savefig(FIGURES_DIR / f"{filename}.svg", bbox_inches="tight")
    fig.savefig(FIGURES_DIR / f"{filename}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def orbital_mean_motion() -> float:
    return math.sqrt(MU_EARTH_KM3_S2 / ORBIT_RADIUS_KM**3)


def make_grid(step_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """生成含边界的规则网格、单位向量和面积权重。"""
    latitudes = np.arange(LAT_RANGE[0], LAT_RANGE[1] + 0.5 * step_deg, step_deg)
    if latitudes[-1] < LAT_RANGE[1] - 1e-9:
        latitudes = np.append(latitudes, LAT_RANGE[1])
    else:
        latitudes[-1] = LAT_RANGE[1]
    longitudes = np.arange(LON_RANGE[0], LON_RANGE[1] + 0.5 * step_deg, step_deg)
    if longitudes[-1] < LON_RANGE[1] - 1e-9:
        longitudes = np.append(longitudes, LON_RANGE[1])
    else:
        longitudes[-1] = LON_RANGE[1]
    lon_mesh, lat_mesh = np.meshgrid(longitudes, latitudes)
    lat_rad = np.radians(lat_mesh.ravel())
    lon_rad = np.radians(lon_mesh.ravel())
    vectors = np.column_stack(
        [np.cos(lat_rad) * np.cos(lon_rad), np.cos(lat_rad) * np.sin(lon_rad), np.sin(lat_rad)]
    ).astype(np.float32)
    weights = np.cos(lat_rad)
    weights /= weights.sum()
    return latitudes, longitudes, vectors, weights


def satellite_unit_vectors(
    config: Constellation, times_s: np.ndarray
) -> np.ndarray:
    """返回 ECEF 单位位置向量，形状为 (time, satellite, xyz)。"""
    m = np.arange(config.planes, dtype=float)[:, None]
    k = np.arange(config.sats_per_plane, dtype=float)[None, :]
    omega = math.radians(config.raan0_deg) + 2 * math.pi * m / config.planes
    base_phase = (
        math.radians(config.phase0_deg)
        + 2 * math.pi * k / config.sats_per_plane
        + 2 * math.pi * config.phase_factor * m / (config.planes * config.sats_per_plane)
    )
    inc = math.radians(config.inclination_deg)
    u = orbital_mean_motion() * times_s[:, None, None] + base_phase[None, :, :]
    cos_u, sin_u = np.cos(u), np.sin(u)
    cos_o, sin_o = np.cos(omega)[None, :, :], np.sin(omega)[None, :, :]
    x_i = cos_o * cos_u - sin_o * sin_u * math.cos(inc)
    y_i = sin_o * cos_u + cos_o * sin_u * math.cos(inc)
    z_i = np.broadcast_to(sin_u * math.sin(inc), x_i.shape)
    theta = OMEGA_EARTH_RAD_S * times_s[:, None, None]
    x_e = np.cos(theta) * x_i + np.sin(theta) * y_i
    y_e = -np.sin(theta) * x_i + np.cos(theta) * y_i
    xyz = np.stack([x_e, y_e, z_i], axis=-1)
    return xyz.reshape(len(times_s), config.total_satellites, 3).astype(np.float32)


def coverage_counts(
    config: Constellation,
    times_s: np.ndarray,
    ground_vectors: np.ndarray,
    psi_rad: float = PSI_STRICT_RAD,
    time_chunk: int = 8,
    grid_chunk: int = 512,
) -> np.ndarray:
    """分块计算每个时刻、每个网格点的覆盖重数。"""
    counts = np.empty((len(times_s), len(ground_vectors)), dtype=np.int16)
    threshold = math.cos(psi_rad)
    for t0 in range(0, len(times_s), time_chunk):
        t1 = min(t0 + time_chunk, len(times_s))
        sat = satellite_unit_vectors(config, times_s[t0:t1])
        for g0 in range(0, len(ground_vectors), grid_chunk):
            g1 = min(g0 + grid_chunk, len(ground_vectors))
            dots = np.einsum("tsc,gc->tsg", sat, ground_vectors[g0:g1], optimize=True)
            counts[t0:t1, g0:g1] = np.count_nonzero(dots >= threshold, axis=1)
    return counts


def max_zero_gap(mask: np.ndarray, dt_s: float) -> float:
    """计算所有位置中最长连续 False 时段，时间维按周期闭合。"""
    longest = 0
    for column in mask.T:
        false = ~column
        if not np.any(false):
            continue
        doubled = np.concatenate([false, false])
        changes = np.diff(np.concatenate([[0], doubled.view(np.int8), [0]]))
        starts = np.flatnonzero(changes == 1)
        ends = np.flatnonzero(changes == -1)
        longest = max(longest, min(len(column), int(np.max(ends - starts))))
    return longest * dt_s


def evaluate_counts(
    counts: np.ndarray, weights: np.ndarray, dt_s: float, psi_rad: float
) -> tuple[CoverageMetrics, dict[str, np.ndarray]]:
    single = counts >= 1
    double = counts >= 2
    instantaneous_single = single @ weights
    instantaneous_double = double @ weights
    point_single = single.mean(axis=0)
    point_double = double.mean(axis=0)
    worst_idx = int(np.argmin(instantaneous_single))
    min_count = int(counts.min())
    # 覆盖裕量在仅有重数数据时以 0/负值占位；精细候选另算角度裕量。
    min_margin_deg = 0.0 if min_count >= 1 else -math.degrees(psi_rad)
    metrics = CoverageMetrics(
        min_multiplicity=min_count,
        spacetime_single_ratio=float(np.average(instantaneous_single)),
        min_point_single_availability=float(point_single.min()),
        spacetime_double_ratio=float(np.average(instantaneous_double)),
        min_point_double_availability=float(point_double.min()),
        max_gap_s=max_zero_gap(single, dt_s),
        worst_time_index=worst_idx,
        worst_time_single_ratio=float(instantaneous_single[worst_idx]),
        average_multiplicity=float(np.average(counts @ weights)),
        min_margin_deg=min_margin_deg,
    )
    arrays = {
        "instantaneous_single": instantaneous_single,
        "instantaneous_double": instantaneous_double,
        "point_single": point_single,
        "point_double": point_double,
    }
    return metrics, arrays


def evaluate_configuration(
    config: Constellation,
    times_s: np.ndarray,
    ground_vectors: np.ndarray,
    weights: np.ndarray,
    psi_rad: float = PSI_STRICT_RAD,
) -> tuple[CoverageMetrics, np.ndarray, dict[str, np.ndarray]]:
    counts = coverage_counts(config, times_s, ground_vectors, psi_rad=psi_rad)
    dt = float(times_s[1] - times_s[0]) if len(times_s) > 1 else SIDEREAL_DAY_S
    metrics, arrays = evaluate_counts(counts, weights, dt, psi_rad)
    return metrics, counts, arrays


def screening_points() -> tuple[np.ndarray, np.ndarray]:
    latitudes = np.array([4.0, 15.0, 30.0, 45.0, 53.0])
    longitudes = np.linspace(73.0, 135.0, 8)
    lon_mesh, lat_mesh = np.meshgrid(longitudes, latitudes)
    lat = np.radians(lat_mesh.ravel())
    lon = np.radians(lon_mesh.ravel())
    vectors = np.column_stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)]).astype(np.float32)
    weights = np.cos(lat)
    weights /= weights.sum()
    return vectors, weights


def phase_candidates(planes: int) -> list[int]:
    return sorted(set([0, 1, planes // 4, planes // 2, (3 * planes) // 4, planes - 1]))


def hierarchical_search() -> pd.DataFrame:
    """先筛查后复核；结果是规则 Walker 参数空间内的数值候选。"""
    screen_vectors, screen_weights = screening_points()
    screen_times = np.arange(0.0, SIDEREAL_DAY_S, 1800.0)
    coarse_lat, coarse_lon, coarse_vectors, coarse_weights = make_grid(3.0)
    del coarse_lat, coarse_lon
    coarse_times = np.arange(0.0, SIDEREAL_DAY_S, 600.0)

    candidate_pairs = sorted(
        [(m * n, m, n) for m in range(28, 47) for n in range(30, 49)],
        key=lambda x: (x[0], x[1]),
    )
    rows: list[dict[str, float | int | str]] = []
    screen_survivors: list[tuple[float, Constellation]] = []
    inclinations = [49.0, 52.0, 55.0, 58.0, 60.0]

    for total, planes, sats_per_plane in candidate_pairs:
        for inc in inclinations:
            for phase_factor in phase_candidates(planes):
                cfg = Constellation(planes, sats_per_plane, inc, phase_factor)
                counts = coverage_counts(cfg, screen_times, screen_vectors, time_chunk=6, grid_chunk=64)
                single_ratio = float((counts >= 1).mean())
                double_ratio = float((counts >= 2).mean())
                min_count = int(counts.min())
                score = single_ratio + 0.25 * double_ratio + 0.05 * min_count - total / 100000.0
                if single_ratio >= 0.985:
                    screen_survivors.append((score, cfg))

        # 一旦在较小总星数附近积累足够候选，不再扩张过远。
        if len(screen_survivors) >= 80:
            smallest = min(cfg.total_satellites for _, cfg in screen_survivors)
            if total > smallest + 180:
                break

    screen_survivors.sort(key=lambda item: (-item[0], item[1].total_satellites))
    unique: dict[Constellation, float] = {}
    for score, cfg in screen_survivors:
        unique.setdefault(cfg, score)
    selected = sorted(unique.items(), key=lambda item: (item[0].total_satellites, -item[1]))[:120]
    # 高密度边界锚点：用于防止低总星数优先截断漏掉首次可行区。
    anchors = [
        Constellation(38, 52, 50.0, 19),
        Constellation(40, 52, 50.0, 20),
        Constellation(44, 48, 50.0, 0),
    ]
    selected_configs = {cfg for cfg, _ in selected}
    for cfg in anchors:
        if cfg not in selected_configs:
            selected.append((cfg, 0.0))

    for cfg, screen_score in selected:
        metrics, _, _ = evaluate_configuration(cfg, coarse_times, coarse_vectors, coarse_weights)
        row = {
            **asdict(cfg),
            "total_satellites": cfg.total_satellites,
            "screen_score": screen_score,
            **asdict(metrics),
            "evaluation": "coarse_3deg_600s",
        }
        rows.append(row)

    result = pd.DataFrame(rows)
    if result.empty:
        raise RuntimeError("筛查未找到候选，请扩大搜索范围")
    return result.sort_values(
        ["min_multiplicity", "min_point_single_availability", "total_satellites"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def choose_candidates(search_df: pd.DataFrame) -> tuple[Constellation, Constellation]:
    single_pool = search_df[search_df["min_multiplicity"] >= 1]
    if single_pool.empty:
        # 若粗网格无严格可行解，选择最接近者进入相位精化。
        single_row = search_df.sort_values(
            ["min_point_single_availability", "total_satellites"], ascending=[False, True]
        ).iloc[0]
    else:
        single_row = single_pool.sort_values(
            ["total_satellites", "average_multiplicity"], ascending=[True, False]
        ).iloc[0]

    double_pool = search_df[search_df["min_point_double_availability"] >= 0.95]
    if double_pool.empty:
        double_row = search_df.sort_values(
            ["min_point_double_availability", "total_satellites"], ascending=[False, True]
        ).iloc[0]
    else:
        double_row = double_pool.sort_values(
            ["total_satellites", "min_point_double_availability"], ascending=[True, False]
        ).iloc[0]

    def from_row(row: pd.Series) -> Constellation:
        return Constellation(
            int(row["planes"]),
            int(row["sats_per_plane"]),
            float(row["inclination_deg"]),
            int(row["phase_factor"]),
            float(row["raan0_deg"]),
            float(row["phase0_deg"]),
        )

    return from_row(single_row), from_row(double_row)


def local_phase_refinement(
    base: Constellation,
    times_s: np.ndarray,
    vectors: np.ndarray,
    weights: np.ndarray,
    objective: str,
) -> tuple[Constellation, CoverageMetrics]:
    """在一个轨道面和一个同轨间隔的对称域内精化全局相位。"""
    raan_span = 360.0 / base.planes
    phase_span = 360.0 / base.sats_per_plane
    candidates: list[tuple[tuple[float, ...], Constellation, CoverageMetrics]] = []
    for raan0 in np.linspace(0.0, raan_span, 4, endpoint=False):
        for phase0 in np.linspace(0.0, phase_span, 4, endpoint=False):
            cfg = Constellation(
                base.planes,
                base.sats_per_plane,
                base.inclination_deg,
                base.phase_factor,
                float(raan0),
                float(phase0),
            )
            metrics, _, _ = evaluate_configuration(cfg, times_s, vectors, weights)
            if objective == "single":
                rank = (
                    metrics.min_multiplicity,
                    metrics.min_point_single_availability,
                    metrics.worst_time_single_ratio,
                    metrics.average_multiplicity,
                )
            else:
                rank = (
                    metrics.min_point_double_availability,
                    metrics.spacetime_double_ratio,
                    metrics.min_multiplicity,
                    metrics.average_multiplicity,
                )
            candidates.append((rank, cfg, metrics))
    _, config, metrics = max(candidates, key=lambda item: item[0])
    return config, metrics


def deployment_cost(total_satellites: int) -> tuple[int, float]:
    launches = math.ceil(total_satellites / 60)
    cost = 5e6 * total_satellites + 2e8 * launches
    return launches, cost


def generate_figures(
    search_df: pd.DataFrame,
    single_cfg: Constellation,
    double_cfg: Constellation,
    single_counts: np.ndarray,
    double_counts: np.ndarray,
    single_arrays: dict[str, np.ndarray],
    double_arrays: dict[str, np.ndarray],
    times_s: np.ndarray,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    single_metrics: CoverageMetrics,
    double_metrics: CoverageMetrics,
) -> list[tuple[str, str]]:
    figures: list[tuple[str, str]] = []

    # 搜索地形
    fig, ax = plt.subplots(figsize=(8, 6))
    color = search_df["min_point_single_availability"] * 100
    size = 18 + 90 * search_df["min_point_double_availability"].clip(0, 1)
    sc = ax.scatter(search_df["planes"], search_df["sats_per_plane"], c=color, s=size, cmap="viridis", alpha=0.75, edgecolor="white", linewidth=0.3)
    ax.scatter([single_cfg.planes], [single_cfg.sats_per_plane], marker="*", s=170, color=PALETTE["red_strong"], label="单重候选")
    ax.scatter([double_cfg.planes], [double_cfg.sats_per_plane], marker="D", s=55, color=PALETTE["blue_main"], label="二重候选")
    ax.set(xlabel="轨道面数 M", ylabel="每轨卫星数 N")
    ax.legend()
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("最差点单重覆盖时间比例 (%)")
    ax.text(-0.08, 1.06, "a", transform=ax.transAxes, fontsize=22, fontweight="bold", va="top", ha="right")
    fig.tight_layout()
    save_figure(fig, "q2_search_landscape")
    figures.append(("q2_search_landscape", "星座搜索可行域"))

    # 最差时刻覆盖图
    worst_map = single_counts[single_metrics.worst_time_index].reshape(len(latitudes), len(longitudes))
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(worst_map, origin="lower", extent=[longitudes[0], longitudes[-1], latitudes[0], latitudes[-1]], aspect="auto", cmap="Blues", vmin=0)
    ax.set(xlabel="经度 (°E)", ylabel="纬度 (°N)")
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("覆盖重数")
    ax.text(0.02, 0.98, f"t = {times_s[single_metrics.worst_time_index]/3600:.2f} h", transform=ax.transAxes, va="top", color=PALETTE["neutral_mid"])
    ax.text(-0.08, 1.06, "a", transform=ax.transAxes, fontsize=22, fontweight="bold", va="top", ha="right")
    fig.tight_layout()
    save_figure(fig, "q2_worst_coverage_map")
    figures.append(("q2_worst_coverage_map", "最差时刻覆盖重数"))

    # 时间序列
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(times_s / 3600, single_counts.min(axis=1), color=PALETTE["blue_main"], label="单重候选")
    ax1.plot(times_s / 3600, double_counts.min(axis=1), color=PALETTE["teal"], alpha=0.85, label="二重候选")
    ax1.axhline(1, color=PALETTE["neutral_mid"], ls="--")
    ax1.set(xlabel="时间 (h)", ylabel="区域最小覆盖重数")
    ax1.legend()
    ax1.text(-0.08, 1.08, "a", transform=ax1.transAxes, fontsize=22, fontweight="bold", va="top", ha="right")
    ax2.plot(times_s / 3600, 100 * single_arrays["instantaneous_double"], color=PALETTE["blue_secondary"], label="单重候选")
    ax2.plot(times_s / 3600, 100 * double_arrays["instantaneous_double"], color=PALETTE["red_strong"], label="二重候选")
    ax2.axhline(95, color=PALETTE["neutral_mid"], ls="--", label="95%")
    ax2.set(xlabel="时间 (h)", ylabel="区域二重覆盖面积比例 (%)")
    ax2.legend()
    ax2.text(-0.08, 1.08, "b", transform=ax2.transAxes, fontsize=22, fontweight="bold", va="top", ha="right")
    fig.tight_layout()
    save_figure(fig, "q2_temporal_coverage")
    figures.append(("q2_temporal_coverage", "全天覆盖能力"))

    # 规模和成本
    labels = ["单重候选", "二重候选"]
    totals = [single_cfg.total_satellites, double_cfg.total_satellites]
    costs = [deployment_cost(v)[1] / 1e8 for v in totals]
    launches = [deployment_cost(v)[0] for v in totals]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    bars = ax1.bar(labels, totals, color=[PALETTE["blue_main"], PALETTE["red_strong"]])
    ax1.set_ylabel("卫星总数 (颗)")
    for bar, value in zip(bars, totals):
        ax1.text(bar.get_x() + bar.get_width()/2, value, str(value), ha="center", va="bottom")
    ax1.text(-0.08, 1.08, "a", transform=ax1.transAxes, fontsize=22, fontweight="bold", va="top", ha="right")
    bars = ax2.bar(labels, costs, color=[PALETTE["blue_main"], PALETTE["red_strong"]])
    ax2.set_ylabel("部署成本 (亿元)")
    for bar, value, launch in zip(bars, costs, launches):
        ax2.text(bar.get_x() + bar.get_width()/2, value, f"{value:.1f}\n{launch}次发射", ha="center", va="bottom")
    ax2.text(-0.08, 1.08, "b", transform=ax2.transAxes, fontsize=22, fontweight="bold", va="top", ha="right")
    fig.tight_layout()
    save_figure(fig, "q2_scale_cost_comparison")
    figures.append(("q2_scale_cost_comparison", "规模与成本对比"))
    return figures


def update_html_panel(figure_list: list[tuple[str, str]]) -> None:
    panel = FIGURES_DIR / "图表面板.html"
    existing = ""
    if panel.exists():
        existing = panel.read_text(encoding="utf-8")
    cards = "".join(
        f'<section><h2>{html.escape(title)}</h2><object data="{html.escape(name)}.svg" type="image/svg+xml"></object></section>'
        for name, title in figure_list
    )
    if existing and "</main>" in existing:
        existing = existing.replace("</main>", cards + "</main>")
        panel.write_text(existing, encoding="utf-8")
    else:
        panel.write_text(
            f"<!doctype html><meta charset='utf-8'><main>{cards}</main>", encoding="utf-8"
        )


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    configure_plots()

    print("阶段 1/3：规则 Walker 参数筛查")
    search_df = hierarchical_search()
    search_df.to_csv(RESULTS_DIR / "问题2_候选搜索结果.csv", index=False, encoding="utf-8-sig")
    base_single, base_double = choose_candidates(search_df)

    print("阶段 2/3：初始相位精化")
    refine_lat, refine_lon, refine_vectors, refine_weights = make_grid(2.0)
    refine_times = np.arange(0.0, SIDEREAL_DAY_S, 300.0)
    single_cfg, _ = local_phase_refinement(base_single, refine_times, refine_vectors, refine_weights, "single")
    double_cfg, _ = local_phase_refinement(base_double, refine_times, refine_vectors, refine_weights, "double")

    print("阶段 3/3：1°、120 s 独立复核")
    fine_lat, fine_lon, fine_vectors, fine_weights = make_grid(1.0)
    fine_times = np.arange(0.0, SIDEREAL_DAY_S, 120.0)
    single_metrics, single_counts, single_arrays = evaluate_configuration(single_cfg, fine_times, fine_vectors, fine_weights)
    double_metrics, double_counts, double_arrays = evaluate_configuration(double_cfg, fine_times, fine_vectors, fine_weights)

    final_rows = []
    for scenario, cfg, metrics in [
        ("single_candidate", single_cfg, single_metrics),
        ("double_candidate", double_cfg, double_metrics),
    ]:
        launches, cost = deployment_cost(cfg.total_satellites)
        final_rows.append(
            {
                "scenario": scenario,
                **asdict(cfg),
                "total_satellites": cfg.total_satellites,
                **asdict(metrics),
                "launches": launches,
                "deployment_cost_yuan": cost,
                "verification_grid_deg": 1.0,
                "verification_dt_s": 120.0,
                "status_single": metrics.min_multiplicity >= 1,
                "status_double95": metrics.min_point_double_availability >= 0.95,
            }
        )
    final_df = pd.DataFrame(final_rows)
    final_df.to_csv(RESULTS_DIR / "问题2_推荐构型.csv", index=False, encoding="utf-8-sig")

    point_df = pd.DataFrame(
        {
            "latitude_deg": np.repeat(fine_lat, len(fine_lon)),
            "longitude_deg": np.tile(fine_lon, len(fine_lat)),
            "single_time_availability": single_arrays["point_single"],
            "single_double_availability": single_arrays["point_double"],
            "double_time_availability": double_arrays["point_single"],
            "double_double_availability": double_arrays["point_double"],
        }
    )
    point_df.to_csv(RESULTS_DIR / "问题2_空间覆盖指标.csv", index=False, encoding="utf-8-sig")

    figures = generate_figures(
        search_df,
        single_cfg,
        double_cfg,
        single_counts,
        double_counts,
        single_arrays,
        double_arrays,
        fine_times,
        fine_lat,
        fine_lon,
        single_metrics,
        double_metrics,
    )
    update_html_panel(figures)

    lines = [
        "问题二数值候选（规则 Walker 搜索）",
        f"单重候选: M={single_cfg.planes}, N={single_cfg.sats_per_plane}, S={single_cfg.total_satellites}, i={single_cfg.inclination_deg:.1f}°, F={single_cfg.phase_factor}",
        f"  最小覆盖重数={single_metrics.min_multiplicity}, 最差点单重可用度={single_metrics.min_point_single_availability:.6f}",
        f"二重候选: M={double_cfg.planes}, N={double_cfg.sats_per_plane}, S={double_cfg.total_satellites}, i={double_cfg.inclination_deg:.1f}°, F={double_cfg.phase_factor}",
        f"  最差点二重可用度={double_metrics.min_point_double_availability:.6f}, 时空二重覆盖率={double_metrics.spacetime_double_ratio:.6f}",
        "复核尺度: 1°空间网格、120 s 时间步、1个恒星日。",
        "注意：这是给定规则参数范围内的数值候选，不宣称连续时空上的全局数学最优。",
    ]
    (RESULTS_DIR / "问题2_结果说明.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
