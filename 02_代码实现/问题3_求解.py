"""问题三：时变星间链路、端到端时延与容量情景分析。"""

from __future__ import annotations

import html
import math
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra

import 问题2_求解 as q2


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"

C_KM_S = 299792.458
MAX_ISL_KM = 5000.0
PROCESSING_MS = 0.5
ACCESS_CAPACITY_GBPS = 20.0

PALETTE = {
    "blue_main": "#0F4D92",
    "blue_secondary": "#3775BA",
    "red_strong": "#B64342",
    "neutral_mid": "#767676",
    "neutral_light": "#CFCECE",
    "teal": "#42949E",
    "green": "#8BCF8B",
}


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


def load_constellation() -> q2.Constellation:
    path = RESULTS_DIR / "问题2_推荐构型.csv"
    row = pd.read_csv(path).iloc[0]
    return q2.Constellation(
        int(row["planes"]), int(row["sats_per_plane"]), float(row["inclination_deg"]),
        int(row["phase_factor"]), float(row["raan0_deg"]), float(row["phase0_deg"])
    )


def add_edge(edge_map: dict[tuple[int, int], tuple[float, str]], u: int, v: int, distance: float, kind: str) -> None:
    if u == v:
        return
    key = (u, v) if u < v else (v, u)
    old = edge_map.get(key)
    if old is None or distance < old[0]:
        edge_map[key] = (float(distance), kind)


def build_topology(config: q2.Constellation, time_s: float) -> tuple[csr_matrix, pd.DataFrame, np.ndarray]:
    """构建四邻接无向图；相邻轨道面使用最小总距离一一匹配。"""
    unit = q2.satellite_unit_vectors(config, np.array([time_s]))[0].astype(float)
    positions = q2.ORBIT_RADIUS_KM * unit
    m_count, n_count = config.planes, config.sats_per_plane
    edge_map: dict[tuple[int, int], tuple[float, str]] = {}

    # 同轨前后邻居
    intra_distance = 2 * q2.ORBIT_RADIUS_KM * math.sin(math.pi / n_count)
    for m in range(m_count):
        offset = m * n_count
        for k in range(n_count):
            add_edge(edge_map, offset + k, offset + (k + 1) % n_count, intra_distance, "intra")

    # 相邻轨道面一一匹配；每对轨道面只处理一次。
    for m in range(m_count):
        mp = (m + 1) % m_count
        a_idx = np.arange(m * n_count, (m + 1) * n_count)
        b_idx = np.arange(mp * n_count, (mp + 1) * n_count)
        diff = positions[a_idx, None, :] - positions[b_idx][None, :, :]
        cost = np.linalg.norm(diff, axis=2)
        rows, cols = linear_sum_assignment(cost)
        for r, c in zip(rows, cols):
            distance = cost[r, c]
            if distance <= MAX_ISL_KM:
                add_edge(edge_map, int(a_idx[r]), int(b_idx[c]), float(distance), "inter")

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    edge_rows: list[dict[str, float | int | str]] = []
    for (u, v), (distance, kind) in edge_map.items():
        weight_ms = distance / C_KM_S * 1000.0 + PROCESSING_MS
        rows.extend([u, v])
        cols.extend([v, u])
        data.extend([weight_ms, weight_ms])
        edge_rows.append({"u": u, "v": v, "distance_km": distance, "kind": kind, "weight_ms": weight_ms})
    graph = csr_matrix((data, (rows, cols)), shape=(config.total_satellites, config.total_satellites))
    return graph, pd.DataFrame(edge_rows), unit


def ground_sites() -> tuple[pd.DataFrame, np.ndarray]:
    """使用规则区域站点代表均匀业务；保留边界和内部点。"""
    latitudes = np.linspace(q2.LAT_RANGE[0], q2.LAT_RANGE[1], 6)
    longitudes = np.linspace(q2.LON_RANGE[0], q2.LON_RANGE[1], 6)
    lon_mesh, lat_mesh = np.meshgrid(longitudes, latitudes)
    lat = np.radians(lat_mesh.ravel())
    lon = np.radians(lon_mesh.ravel())
    vectors = np.column_stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])
    sites = pd.DataFrame(
        {"site_id": np.arange(len(lat)), "latitude_deg": np.degrees(lat), "longitude_deg": np.degrees(lon)}
    )
    return sites, vectors


def access_sets(unit_sat: np.ndarray, ground_vectors: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
    dots = ground_vectors @ unit_sat.T
    visible: list[np.ndarray] = []
    delays: list[np.ndarray] = []
    threshold = math.cos(q2.PSI_STRICT_RAD)
    for row in dots:
        idx = np.flatnonzero(row >= threshold)
        slant = np.sqrt(
            q2.ORBIT_RADIUS_KM**2 + q2.R_EARTH_KM**2
            - 2 * q2.ORBIT_RADIUS_KM * q2.R_EARTH_KM * row[idx]
        )
        visible.append(idx)
        delays.append(slant / C_KM_S * 1000.0)
    return visible, delays


def reconstruct_path(predecessors: np.ndarray, source_row: int, source: int, target: int) -> list[int]:
    if source == target:
        return [source]
    path = [target]
    current = target
    for _ in range(predecessors.shape[1] + 1):
        current = int(predecessors[source_row, current])
        if current < 0:
            return []
        path.append(current)
        if current == source:
            return path[::-1]
    return []


def route_all_pairs(
    graph: csr_matrix,
    visible: list[np.ndarray],
    access_delay: list[np.ndarray],
    return_paths: bool = False,
) -> tuple[pd.DataFrame, dict[tuple[int, int], list[int]]]:
    sources = np.unique(np.concatenate(visible)).astype(int)
    source_to_row = {int(s): i for i, s in enumerate(sources)}
    if return_paths:
        distances, predecessors = dijkstra(graph, directed=False, indices=sources, return_predecessors=True)
    else:
        distances = dijkstra(graph, directed=False, indices=sources, return_predecessors=False)
        predecessors = None

    rows = []
    paths: dict[tuple[int, int], list[int]] = {}
    for a in range(len(visible)):
        for b in range(a + 1, len(visible)):
            best = (math.inf, -1, -1, -1, -1)
            for ia, src in enumerate(visible[a]):
                dist_row = distances[source_to_row[int(src)], visible[b]]
                totals = access_delay[a][ia] + dist_row + access_delay[b] + PROCESSING_MS
                j = int(np.argmin(totals))
                if totals[j] < best[0]:
                    best = (float(totals[j]), int(src), int(visible[b][j]), ia, j)
            delay, src, dst, _, _ = best
            hops = math.nan
            if return_paths and math.isfinite(delay):
                source_row = source_to_row[src]
                path = reconstruct_path(predecessors, source_row, src, dst)
                paths[(a, b)] = path
                hops = max(0, len(path) - 1)
            rows.append({"site_a": a, "site_b": b, "delay_ms": delay, "source_sat": src, "target_sat": dst, "isl_hops": hops})
    return pd.DataFrame(rows), paths


def topology_and_latency(config: q2.Constellation) -> tuple[pd.DataFrame, pd.DataFrame, dict[float, tuple[csr_matrix, pd.DataFrame, np.ndarray]]]:
    sites, ground_vectors = ground_sites()
    del sites
    times = np.arange(0.0, q2.SIDEREAL_DAY_S, 1800.0)
    topo_rows = []
    delay_frames = []
    snapshots: dict[float, tuple[csr_matrix, pd.DataFrame, np.ndarray]] = {}
    for time_s in times:
        graph, edges, unit = build_topology(config, float(time_s))
        n_components, _ = connected_components(graph, directed=False)
        degrees = np.diff(graph.indptr)
        inter = edges[edges.kind == "inter"].distance_km
        topo_rows.append(
            {
                "time_s": time_s,
                "edges": len(edges),
                "components": n_components,
                "min_degree": int(degrees.min()),
                "max_degree": int(degrees.max()),
                "mean_degree": float(degrees.mean()),
                "inter_min_km": float(inter.min()),
                "inter_median_km": float(inter.median()),
                "inter_p95_km": float(inter.quantile(0.95)),
                "inter_max_km": float(inter.max()),
            }
        )
        visible, access_delay = access_sets(unit, ground_vectors)
        pair_df, _ = route_all_pairs(graph, visible, access_delay, return_paths=False)
        pair_df["time_s"] = time_s
        delay_frames.append(pair_df)
        snapshots[float(time_s)] = (graph, edges, unit)
    return pd.DataFrame(topo_rows), pd.concat(delay_frames, ignore_index=True), snapshots


def capacity_scenarios(
    graph: csr_matrix, edges: pd.DataFrame, unit: np.ndarray
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """在最坏时刻按等权OD最短路径分配单位总业务，求容量瓶颈。"""
    _, ground_vectors = ground_sites()
    visible, access_delay = access_sets(unit, ground_vectors)
    pair_df, paths = route_all_pairs(graph, visible, access_delay, return_paths=True)
    valid_pairs = pair_df[np.isfinite(pair_df.delay_ms)]
    demand_per_pair = 1.0 / len(valid_pairs)  # 每 1 Gbps 总业务的分配比例
    edge_load: defaultdict[tuple[int, int], float] = defaultdict(float)
    access_load: defaultdict[int, float] = defaultdict(float)
    for row in valid_pairs.itertuples():
        path = paths[(row.site_a, row.site_b)]
        access_load[int(row.source_sat)] += demand_per_pair
        access_load[int(row.target_sat)] += demand_per_pair
        for u, v in zip(path[:-1], path[1:]):
            key = (u, v) if u < v else (v, u)
            edge_load[key] += demand_per_pair

    max_access_fraction = max(access_load.values())
    access_peak_limit = ACCESS_CAPACITY_GBPS / max_access_fraction
    edge_table = edges.copy()
    edge_table["load_per_unit"] = [edge_load.get((min(int(u), int(v)), max(int(u), int(v))), 0.0) for u, v in zip(edges.u, edges.v)]
    max_edge_fraction = max(edge_load.values())
    rows = []
    for isl_capacity in [10.0, 20.0, 40.0, 80.0, 120.0]:
        isl_peak_limit = isl_capacity / max_edge_fraction
        peak = min(access_peak_limit, isl_peak_limit)
        bottleneck = "access" if access_peak_limit <= isl_peak_limit else "ISL"
        rows.append(
            {
                "isl_capacity_gbps": isl_capacity,
                "access_peak_limit_gbps": access_peak_limit,
                "isl_peak_limit_gbps": isl_peak_limit,
                "system_peak_capacity_gbps": peak,
                "safe_average_capacity_gbps": peak / 1.5,
                "bottleneck": bottleneck,
                "max_access_load_per_unit": max_access_fraction,
                "max_isl_load_per_unit": max_edge_fraction,
            }
        )
    return pd.DataFrame(rows), edge_table


def lon_lat(unit: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lat = np.degrees(np.arcsin(np.clip(unit[:, 2], -1, 1)))
    lon = np.degrees(np.arctan2(unit[:, 1], unit[:, 0]))
    return lon, lat


def generate_figures(
    topology: pd.DataFrame,
    delays: pd.DataFrame,
    capacity: pd.DataFrame,
    snapshot: tuple[csr_matrix, pd.DataFrame, np.ndarray],
) -> list[tuple[str, str]]:
    figures = []
    graph, edges, unit = snapshot
    del graph
    lon, lat = lon_lat(unit)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.scatter(lon, lat, s=3, color=PALETTE["blue_main"], alpha=0.65)
    for row in edges.itertuples():
        if row.kind != "inter":
            continue
        x = [lon[row.u], lon[row.v]]
        if abs(x[0] - x[1]) > 180:
            continue
        ax.plot(x, [lat[row.u], lat[row.v]], color=PALETTE["teal"], lw=0.25, alpha=0.22)
    ax.plot([73, 135, 135, 73, 73], [4, 4, 53, 53, 4], color=PALETTE["red_strong"], lw=1.2)
    ax.set(xlim=(-180, 180), ylim=(-60, 60), xlabel="经度 (°)", ylabel="纬度 (°)")
    ax.text(-0.06, 1.06, "a", transform=ax.transAxes, fontsize=22, fontweight="bold", va="top", ha="right")
    fig.tight_layout()
    save_figure(fig, "q3_topology_snapshot")
    figures.append(("q3_topology_snapshot", "时变星间网络快照"))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(topology.time_s / 3600, topology.inter_median_km, color=PALETTE["blue_main"], label="中位数")
    ax1.plot(topology.time_s / 3600, topology.inter_p95_km, color=PALETTE["teal"], label="95%分位")
    ax1.plot(topology.time_s / 3600, topology.inter_max_km, color=PALETTE["red_strong"], label="最大值")
    ax1.axhline(MAX_ISL_KM, color=PALETTE["neutral_mid"], ls="--", label="5000 km门限")
    ax1.set(xlabel="时间 (h)", ylabel="跨轨链路长度 (km)")
    ax1.legend()
    ax1.text(-0.08, 1.08, "a", transform=ax1.transAxes, fontsize=22, fontweight="bold", va="top", ha="right")
    ax2.plot(topology.time_s / 3600, topology.components, color=PALETTE["blue_secondary"], label="连通分量")
    ax2.plot(topology.time_s / 3600, topology.mean_degree, color=PALETTE["teal"], label="平均度数")
    ax2.set(xlabel="时间 (h)", ylabel="拓扑指标")
    ax2.legend()
    ax2.text(-0.08, 1.08, "b", transform=ax2.transAxes, fontsize=22, fontweight="bold", va="top", ha="right")
    fig.tight_layout()
    save_figure(fig, "q3_link_dynamics")
    figures.append(("q3_link_dynamics", "跨轨链路动态"))

    finite = np.sort(delays.loc[np.isfinite(delays.delay_ms), "delay_ms"].to_numpy())
    cdf = np.arange(1, len(finite) + 1) / len(finite)
    grouped = delays.groupby("time_s").delay_ms.agg(["mean", "max"]).reset_index()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(finite, cdf * 100, color=PALETTE["blue_main"])
    ax1.axvline(30, color=PALETTE["red_strong"], ls="--", label="30 ms")
    ax1.set(xlabel="端到端时延 (ms)", ylabel="累计比例 (%)")
    ax1.legend()
    ax1.text(-0.08, 1.08, "a", transform=ax1.transAxes, fontsize=22, fontweight="bold", va="top", ha="right")
    ax2.plot(grouped.time_s / 3600, grouped["mean"], color=PALETTE["blue_main"], label="平均值")
    ax2.plot(grouped.time_s / 3600, grouped["max"], color=PALETTE["red_strong"], label="最大值")
    ax2.axhline(30, color=PALETTE["neutral_mid"], ls="--")
    ax2.set(xlabel="时间 (h)", ylabel="端到端时延 (ms)")
    ax2.legend()
    ax2.text(-0.08, 1.08, "b", transform=ax2.transAxes, fontsize=22, fontweight="bold", va="top", ha="right")
    fig.tight_layout()
    save_figure(fig, "q3_delay_distribution")
    figures.append(("q3_delay_distribution", "端到端时延分布"))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(capacity.isl_capacity_gbps, capacity.system_peak_capacity_gbps, marker="o", color=PALETTE["blue_main"], label="峰值承载量")
    ax.plot(capacity.isl_capacity_gbps, capacity.safe_average_capacity_gbps, marker="o", color=PALETTE["teal"], label="峰均比1.5下安全均值")
    ax.set(xlabel="单条ISL容量情景 (Gbps)", ylabel="区域样本业务承载量 (Gbps)")
    ax.legend()
    ax.text(-0.06, 1.06, "a", transform=ax.transAxes, fontsize=22, fontweight="bold", va="top", ha="right")
    fig.tight_layout()
    save_figure(fig, "q3_capacity_scenarios")
    figures.append(("q3_capacity_scenarios", "容量情景分析"))
    return figures


def update_html_panel(figure_list: list[tuple[str, str]]) -> None:
    panel = FIGURES_DIR / "图表面板.html"
    text = panel.read_text(encoding="utf-8") if panel.exists() else "<main></main>"
    cards = "".join(
        f'<section><h2>{html.escape(title)}</h2><object data="{html.escape(name)}.svg" type="image/svg+xml"></object></section>'
        for name, title in figure_list
    )
    panel.write_text(text.replace("</main>", cards + "</main>"), encoding="utf-8")


def main() -> None:
    configure_plots()
    config = load_constellation()
    print(f"读取问题二构型: {config}")
    topology, delays, snapshots = topology_and_latency(config)
    topology.to_csv(RESULTS_DIR / "问题3_拓扑时间序列.csv", index=False, encoding="utf-8-sig")
    delays.to_csv(RESULTS_DIR / "问题3_点对时延.csv", index=False, encoding="utf-8-sig")

    time_stats = delays.groupby("time_s").delay_ms.agg(["mean", "max"]).reset_index()
    worst_time = float(time_stats.loc[time_stats["max"].idxmax(), "time_s"])
    capacity, edge_load = capacity_scenarios(*snapshots[worst_time])
    capacity.to_csv(RESULTS_DIR / "问题3_容量情景.csv", index=False, encoding="utf-8-sig")
    edge_load.to_csv(RESULTS_DIR / "问题3_最坏时刻链路负载.csv", index=False, encoding="utf-8-sig")

    finite_delays = delays.delay_ms[np.isfinite(delays.delay_ms)]
    unreachable = int((~np.isfinite(delays.delay_ms)).sum())
    summary = pd.DataFrame(
        [
            {"metric": "min_components", "value": topology.components.min(), "unit": "count"},
            {"metric": "max_components", "value": topology.components.max(), "unit": "count"},
            {"metric": "min_node_degree", "value": topology.min_degree.min(), "unit": "count"},
            {"metric": "max_node_degree", "value": topology.max_degree.max(), "unit": "count"},
            {"metric": "max_interplane_link_km", "value": topology.inter_max_km.max(), "unit": "km"},
            {"metric": "mean_delay_ms", "value": finite_delays.mean(), "unit": "ms"},
            {"metric": "p95_delay_ms", "value": finite_delays.quantile(0.95), "unit": "ms"},
            {"metric": "max_delay_ms", "value": finite_delays.max(), "unit": "ms"},
            {"metric": "delay_30ms_compliance", "value": (finite_delays <= 30).mean(), "unit": "ratio"},
            {"metric": "unreachable_pairs", "value": unreachable, "unit": "count"},
        ]
    )
    summary.to_csv(RESULTS_DIR / "问题3_核心结果.csv", index=False, encoding="utf-8-sig")

    figures = generate_figures(topology, delays, capacity, snapshots[0.0])
    update_html_panel(figures)

    lines = [
        "问题三结果",
        f"全天连通分量范围: {topology.components.min()}–{topology.components.max()}",
        f"节点度数范围: {topology.min_degree.min()}–{topology.max_degree.max()}",
        f"最大跨轨链路: {topology.inter_max_km.max():.2f} km",
        f"平均端到端时延: {finite_delays.mean():.3f} ms",
        f"95%分位时延: {finite_delays.quantile(0.95):.3f} ms",
        f"最大端到端时延: {finite_delays.max():.3f} ms",
        f"30 ms合格率: {(finite_delays <= 30).mean():.4%}",
        f"不可达点对样本: {unreachable}",
        "容量结果为36个区域站点均匀OD、最短路分配下的基准情景；ISL容量不是题面给定值。",
    ]
    (RESULTS_DIR / "问题3_结果说明.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
