from __future__ import annotations

import logging
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pvlib

from pathlib import Path
from datetime import datetime
import re

# from mape_maker.MapeMaker import MapeMaker


def make_pharma_cold_occupancy(
    n_hours: int = 8760,
    people_per_zone: np.ndarray | None = None,
    meta_active: float = 150.0,
    meta_patrol: float = 100.0,
    steps_per_hour: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    生成医药冷库（2~6 °C）的人员占用参数，供 ParameterGenerator 直接使用。

    Parameters
    ----------
    n_hours : int
        全年仿真小时数，默认 8760。
    people_per_zone : array-like of shape (3,), optional
        各区域峰值人数 [办公区, 精储区, 散储区]。
        默认 [1, 1, 2]。
    meta_active : float
        早班在岗的人均代谢率（W/人）。
        冷库场景建议 150 W（防寒服 + 轻劳动），默认 150.0。
    meta_patrol : float
        夜班巡视的人均代谢率（W/人），默认 100.0。
    steps_per_hour : int
        每小时的仿真步数，默认 1（即 1h 步长）。
        15min 步长时传入 4。

    Returns
    -------
    full_occ : np.ndarray, shape (3,)
        传给 ParameterGenerator(full_occ=...) 的各区峰值人数。
    activity_sch : np.ndarray, shape (n_hours * steps_per_hour,)
        传给 ParameterGenerator(activity_sch=...) 的逐步人均代谢率（W/人）。
        Meta = 0 表示该时刻无人，BEAR 内部据此计算人体散热 Occupower。

    Notes
    -----
    时间表规则（0 = 1月1日 00:00，周一起算）：

      工作日：
        00:00–05:59  夜班巡视        meta_patrol W/人
        06:00–07:59  早班前空仓      0 W/人
        08:00–15:59  早班在岗        meta_active W/人
        16:00–21:59  交班后空仓      0 W/人
        22:00–23:59  夜班巡视        meta_patrol W/人

      周末（周六/周日）：
        08:00–15:59  值班（活动强度减半） meta_active * 0.5 W/人
        其余时段      空仓              0 W/人
    """
    if people_per_zone is None:
        people_per_zone = np.array([1, 1, 2], dtype=float)
    full_occ = np.asarray(people_per_zone, dtype=float)

    n_steps = n_hours * steps_per_hour
    sch = np.zeros(n_steps, dtype=float)
    steps_per_day = 24 * steps_per_hour
    for s in range(n_steps):
        day_of_week = (s // steps_per_day) % 7   # 0 = 周一 … 6 = 周日
        hour_of_day = (s % steps_per_day) // steps_per_hour
        is_weekend = day_of_week >= 5

        if s % steps_per_hour == 0:
            if is_weekend:
                if 8 <= hour_of_day < 16:
                    sch[s] = meta_active * 0.5
            else:
                if 8 <= hour_of_day < 16:
                    sch[s] = meta_active
                elif hour_of_day < 6 or hour_of_day >= 22:
                    sch[s] = meta_patrol

    return full_occ, sch


def make_tou_price(n_hours: int = 8760, steps_per_hour: int = 1) -> np.ndarray:
    n_steps = n_hours * steps_per_hour
    lam_full = np.zeros(n_steps, dtype=float)
    for s in range(n_steps):
        hour = (s // steps_per_hour) % 24
        if 9 <= hour < 21:
            lam_full[s] = 0.20
        elif 6 <= hour < 9 or 21 <= hour < 23:
            lam_full[s] = 0.12
        else:
            lam_full[s] = 0.07
    return lam_full


def make_pv_energy(ghi_like: np.ndarray, max_power: float, pv_area: float = 100.0, pv_eta: float = 0.18, pv_pr: float = 0.80) -> np.ndarray:
    e_free_full = np.asarray(ghi_like, dtype=float) * pv_area * pv_eta * pv_pr / max_power
    return np.clip(e_free_full, 0.0, 1.0)


def call_grid_power(p_load: float, e_pv: float, lam: float = 1.0, eta_c: float = 1.0, eta_h: float = 1.0, MAX_POWER=8000, STEPS_PER_HOUR=4) -> tuple[float, float]:
    e_load = max(float(p_load), 0.0) / eta_c + max(-float(p_load), 0.0) / eta_h
    e_grid = max(e_load - float(e_pv), 0.0)
    price = e_grid * lam * MAX_POWER / STEPS_PER_HOUR
    return float(e_grid), float(price)


def my_forecast(true_data, std: float = 0.5, effect_rate: float = 0.3, not_negative: bool = False) -> np.ndarray:
    true_data = np.asarray(true_data, dtype=float)
    n_steps = len(true_data)
    predicted_data = np.zeros(n_steps, dtype=float)
    predicted_data[0] = true_data[0]
    perturbation = 0.0
    for i in range(1, n_steps):
        perturbation = effect_rate * perturbation + np.random.normal(0.0, std)
        predicted_data[i] = true_data[i] + perturbation
        if not_negative and (true_data[i] == 0 or predicted_data[i] < 0):
            predicted_data[i] = 0.0
    return predicted_data


def parameter_identification(env, epw_path: str | Path, zone: int = 0, figure_path: str | Path = "building_parameter_identification.png") -> np.ndarray:
    env.reset()

    action_list = np.arange(0, 1, 0.01)
    action_list = np.vstack((action_list, action_list, action_list)).T
    for a in action_list:
        env.step(a)

    controller_state = np.array(env.statelist)
    controller_action = np.array(env.actionlist)

    new_temp = controller_state[1:, zone]
    temp = controller_state[:-1, zone]
    out_temp = controller_state[:-1, 3]
    action = controller_action[:-1, zone]
    # GHI 直接从 env 的状态列表中取（索引 roomnum+1 起的第一列），
    # 避免与 EPW 小时级数据长度不匹配。
    roomnum = env.roomnum
    ghi_used = controller_state[:-1, roomnum + 1]

    y = new_temp
    X = np.column_stack([temp, out_temp, ghi_used, action])
    theta = np.linalg.inv(X.T @ X) @ X.T @ y
    y_pred = X @ theta

    plt.figure(figsize=(10, 6))
    x_axis = range(len(y))
    plt.scatter(x_axis, y, label="Observe data", alpha=0.7)
    plt.plot(x_axis, y, "r-", label="Real environment")
    plt.plot(x_axis, y_pred, "g--", label="Approximate model")
    plt.legend(loc="best", fontsize=22)
    plt.xlabel("x", fontsize=22)
    plt.ylabel("y", fontsize=22)
    plt.title("Least Squares Linear Regression", fontsize=22)
    plt.savefig(str(figure_path), dpi=300, bbox_inches="tight")
    plt.close()

    return theta


def update_h(x: float, i: int, win: int, alpha: float, gama: float, h: float, vt_hist: np.ndarray, yt_hist: np.ndarray, h_delta: float =0.0, TU: float = 6.0, TL: float = 2.0):
    yt = 0.0 if i == 0 else yt_hist[i - 1]
    vt = 1.0 if (x >= TU or x <= TL) else 0.0

    yt = (i * yt + vt) / (i + 1)
    yt_hist[i] = yt
    vt_hist[i] = vt

    if i >= win:
        ywin = vt_hist[i - win + 1 : i + 1].mean()
        h = h * (1 - (alpha - ywin + vt_hist[i - win] / win - 1 / (2 * win)) / gama)
    else:
        ywin = yt
        h = h * (1 - (alpha - ywin + (2 * alpha - 1) / (2 * i + 2)) / gama)

    h = min(h+h_delta, (TU - TL) / 2 - 0.01)-h_delta
    h = max(h, 0.01)
    return float(h), yt_hist, vt_hist


def update_h_oasmpc(x: float, i: int, alpha: float, gama: float, h: float, vt_hist: np.ndarray, yt_hist: np.ndarray, h_delta: float = 0.0, TU: float = 6.0, TL: float = 2.0):
    yt = 0.0 if i == 0 else yt_hist[i - 1]
    vt = 1.0 if (x >= TU or x <= TL) else 0.0

    yt = (i * yt + vt) / (i + 1)
    yt_hist[i] = yt
    vt_hist[i] = vt

    h = h * (1 - (alpha - yt + (2 * alpha - 1) / (2 * i + 2)) / gama)
    h = min(h + h_delta, (TU - TL) / 2 - 0.01) - h_delta
    h = max(h, 0.01)
    return float(h), yt_hist, vt_hist


def compute_energy_breakdown(u: np.ndarray, e_grid: np.ndarray, e_free_true: np.ndarray, eta_c: float = 2.0, eta_h: float = 1.0):
    e_u = np.maximum(u, 0.0) / eta_h + np.maximum(-u, 0.0) / eta_c
    e_pv = np.maximum(e_u - e_grid, 0.0)
    pv_all = np.sum(e_free_true[: len(u)])
    pv_used = np.sum(e_pv)
    e_all = np.sum(e_u)
    self_consumption = pv_used / pv_all if pv_all > 0 else np.nan
    self_sufficiency = pv_used / e_all if e_all > 0 else np.nan
    return e_u, e_pv, float(self_consumption), float(self_sufficiency)


def build_case_dataframe(u: np.ndarray, x: np.ndarray, e_grid: np.ndarray, e_free_true: np.ndarray, h_like: np.ndarray, T_min: float, T_max: float, eta_c: float = 2.0, eta_h: float = 1.0) -> pd.DataFrame:
    e_u, e_pv, _, _ = compute_energy_breakdown(u, e_grid, e_free_true, eta_c=eta_c, eta_h=eta_h)
    v = ((x > T_max) | (x < T_min)).astype(float)

    yt = np.zeros(len(u), dtype=float)
    ywt = np.zeros(len(u), dtype=float)
    d = np.zeros(len(u), dtype=float)
    for i in range(len(u)):
        yt[i] = np.mean(v[: i + 1])
        if i < 499:
            ywt[i] = yt[i]
        else:
            ywt[i] = np.mean(v[i - 499 : i + 1])
        d[i] = max(x[i] - T_max, T_min - x[i], 0.0)

    return pd.DataFrame(
        {
            "u": u,
            "x": x,
            "e_cost": e_u,
            "e_t": e_free_true[: len(u)],
            "e_pvuse": e_pv,
            "e_grid": e_grid,
            "v": v,
            "yt": yt,
            "ywt": ywt,
            "h": h_like,
            "d": d,
        }
    )


def check_performance_indicators(
    df: pd.DataFrame,
    alpha: float = 0.1,
    K_hours: int = 2000,
    eps: float = 0.02
):
    v = np.asarray(df.v, dtype=float)
    d = np.asarray(df.d, dtype=float)
    yt = np.asarray(df.yt, dtype=float)

    p_vio = float(yt[-1])
    e = float(np.mean(d))
    e_max = float(np.max(d))

    in_band = np.abs(yt - alpha) <= eps

    if K_hours == 0:
        # 找到首次进入区间后，后续所有时刻都始终保持在区间内的下标
        # suffix_ok[i] = True 表示从 i 到最后所有 yt 都在区间内
        suffix_ok = np.logical_and.accumulate(in_band[::-1])[::-1]
        valid_start = np.where(suffix_ok)[0]
        t_idx = len(yt) if len(valid_start) == 0 else int(valid_start[0])

    else:
        # 保持原有功能：首次连续 K_hours 个时刻都在区间内
        window_len = int(np.ceil(K_hours))

        if window_len > len(yt):
            t_idx = len(yt)
        else:
            counts = np.convolve(
                in_band.astype(int),
                np.ones(window_len, dtype=int),
                mode="valid"
            )
            valid_start = np.where(counts == window_len)[0]
            t_idx = len(yt) if len(valid_start) == 0 else int(valid_start[0])

    max_len = 0
    curr_len = 0
    best_start = None
    curr_start = None

    for i, val in enumerate(v):
        if val == 1:
            if curr_len == 0:
                curr_start = i
            curr_len += 1
            if curr_len > max_len:
                max_len = curr_len
                best_start = curr_start
        else:
            curr_len = 0
            curr_start = None

    return p_vio, e, e_max, t_idx, int(max_len), best_start


# def train_mape_maker(historical_df: pd.DataFrame, ending_feature: str = "forecasts", a: int = 4, base_process: str = "ARMA"):
#     tmp_csv = "tmp_historical.csv"
#     historical_df.to_csv(tmp_csv, index=True)
#
#     logger = logging.getLogger("mape-maker")
#     logger.setLevel(logging.CRITICAL)
#
#     with warnings.catch_warnings():
#         warnings.simplefilter("ignore")
#         mm = MapeMaker(
#             logger=logger,
#             xyid_path=tmp_csv,
#             ending_feature=ending_feature,
#             a=a,
#             base_process=base_process,
#         )
#
#     logger.setLevel(logging.INFO)
#     r_max = mm.xyid.dataset_info.get("r_m_max")
#     r_max_str = f"{r_max:.0%}" if r_max is not None else "N/A"
#     print(f"MapeMaker trained: ending_feature='{ending_feature}', max MAPE attainable={r_max_str}")
#     return mm


def forecast_solar_daily(mm, actuals_series: pd.Series, start_day: str, end_day: str, n: int = 4, seed: int = 1234, r_tilde: float = 0.20) -> pd.DataFrame:
    days = pd.date_range(start=start_day, end=end_day, freq="D")
    all_results = []

    for day in days:
        day_data = actuals_series[actuals_series.index.date == day.date()]
        daytime = day_data[day_data > 1]

        if len(daytime) == 0:
            zero_df = pd.DataFrame(0.0, index=day_data.index, columns=[f"forecasts_n_{i}" for i in range(n)])
            all_results.append(zero_df)
            continue

        tmp_csv = "tmp_day.csv"
        daytime_df = daytime.to_frame(name="actuals")
        daytime_df.index.name = "datetime"
        daytime_df.to_csv(tmp_csv)

        mm.logger.setLevel(logging.CRITICAL)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                day_results = mm.simulate(
                    sid_file_path=tmp_csv,
                    n=n,
                    seed=seed,
                    r_tilde=r_tilde,
                    curvature_parameters=None,
                )
        except Exception as e:
            print(f"  [WARNING] {day.date()} 预测失败: {e}，用 actual 填充")
            day_results = pd.DataFrame({f"forecasts_n_{i}": daytime.values for i in range(n)}, index=daytime.index)
        finally:
            mm.logger.setLevel(logging.INFO)

        full_day = pd.DataFrame(0.0, index=day_data.index, columns=day_results.columns)
        full_day.loc[day_results.index] = day_results.values
        all_results.append(full_day)
        print(f"  {day.date()} done")

    return pd.concat(all_results)


def forecast_plot(actual_series: pd.Series, results_long: pd.DataFrame):
    actual_plot = actual_series.loc[results_long.index]
    forecast_mean = results_long.mean(axis=1)
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(actual_plot.index, actual_plot.values, label="Actual", color="black", lw=1.5)
    ax.plot(forecast_mean.index, forecast_mean.values, label="Forecast (mean)", color="steelblue", lw=1.5)
    ax.fill_between(
        results_long.index,
        results_long.min(axis=1),
        results_long.max(axis=1),
        alpha=0.2,
        color="steelblue",
        label="Forecast range",
    )
    ax.set_xlabel("Datetime")
    ax.set_ylabel("GHI (W/m²)")
    ax.set_title("Solar Forecast - Daily MapeMaker")
    ax.legend()
    plt.tight_layout()
    plt.show()


def _sanitize_filename(s: str) -> str:
    """将字符串处理成适合作为文件名的形式"""
    s = str(s).strip().replace(" ", "_")
    s = re.sub(r'[\/:*?"<>|]+', "-", s)
    return s


_SUMMARY_LEGACY_COLUMNS_TO_DROP = [
    "violation_rate_empirical",
    "violation_rate_final",
]


def build_summary_from_df(
    df: pd.DataFrame,
    method: str,
    weather: str,
    location: str,
    extra_metrics: dict | None = None,
) -> dict:
    """
    从单个案例的时间序列 df 中提取 summary_results 的一行。

    你后续若想修改 summary_results 的列结构，直接编辑下面
    “summary_items.append(...)” 这一段即可，结构会比较直观。

    Parameters
    ----------
    df : pd.DataFrame
        单个方法的时间序列结果。
    method, weather, location : str
        用于标识案例。
    extra_metrics : dict, optional
        由主程序额外传入的标量结果，例如 total_cost、t_idx、Dmax_hours 等。

    Returns
    -------
    dict
        按插入顺序组织的一行 summary 数据。
    """
    summary_items: list[tuple[str, object]] = [
        ("method", method),
        ("weather", weather),
        ("location", location),
        ("n_steps", len(df)),
        ("updated_at", datetime.now().strftime("%m-%d %H:%M")),
    ]

    p_vio, e, e_max, t_idx, Dmax_hours, start_idx = check_performance_indicators(df, alpha=0.1)

    # ===== 在这里清晰定义你想保留的 summary 字段 =====
    if "cost" in df.columns:
        summary_items.append(("total_energy_use", float(df["e_cost"].sum())))

    if "v" in df.columns:
        summary_items.append(("violation_count", int(df["v"].sum())))

    if "yt" in df.columns:
        summary_items.append(("violation_rate", float(df["yt"].iloc[-1])))
    elif "v" in df.columns:
        summary_items.append(("violation_rate", float(df["v"].mean())))

    if "ywt" in df.columns:
        summary_items.append(("window_violation_rate_final", float(df["ywt"].iloc[-1])))

    summary_items.append(("Stable_t_idx", int(t_idx)))

    summary_items.append(("Max_Vio_H", int(Dmax_hours)))

    if "e_t" in df.columns:
        total_pv_available = float(df["e_t"].sum())
        summary_items.append(("total_pv_available", total_pv_available))
        if total_pv_available > 1e-12 and "e_pvuse" in df.columns:
            summary_items.append(("self_consumption", float(df["e_pvuse"].sum() / total_pv_available)))

    total_energy_use = float(df["e_cost"].sum()) if "e_cost" in df.columns else None
    if total_energy_use is not None and total_energy_use > 1e-12 and "e_pvuse" in df.columns:
        summary_items.append(("self_sufficiency", float(df["e_pvuse"].sum() / total_energy_use)))


    if "d" in df.columns:
        summary_items.append(("mean_violation_magnitude", float(df["d"].mean())))
        summary_items.append(("max_violation_magnitude", float(df["d"].max())))

    if "u" in df.columns:
        summary_items.append(("mean_abs_u", float(np.abs(df["u"]).mean())))
        summary_items.append(("max_abs_u", float(np.abs(df["u"]).max())))

    if "x" in df.columns:
        summary_items.append(("mean_x", float(df["x"].mean())))
        summary_items.append(("min_x", float(df["x"].min())))
        summary_items.append(("max_x", float(df["x"].max())))

    if "e_grid" in df.columns:
        summary_items.append(("total_grid_energy", float(df["e_grid"].sum())))

    if "e_pvuse" in df.columns:
        summary_items.append(("total_pv_used", float(df["e_pvuse"].sum())))

    row = {key: value for key, value in summary_items}

    # 外部额外传入的标量结果放在最后，且允许覆盖同名字段。
    # 例如你在 main_3 中传入 total_cost 时，就会直接写入/覆盖该列。
    if extra_metrics is not None:
        for key, value in extra_metrics.items():
            row[key] = value

    return row


def _append_summary_row_preserve_runtime_order(
    summary_df: pd.DataFrame,
    new_row_df: pd.DataFrame,
    method: str,
    weather: str,
    location: str,
) -> pd.DataFrame:
    """
    按代码实际运行顺序追加一条 summary。

    若同一 (method, weather, location) 已存在，则先删除旧行，
    再把新行追加到最后一行，从而保持“最近一次调用 save_case_output 的顺序”。
    """
    summary_df = summary_df.copy()

    if {"method", "weather", "location"}.issubset(summary_df.columns):
        mask = (
            (summary_df["method"] == method)
            & (summary_df["weather"] == weather)
            & (summary_df["location"] == location)
        )
        summary_df = summary_df.loc[~mask].copy()

    existing_cols = list(summary_df.columns)
    new_cols = [c for c in new_row_df.columns if c not in existing_cols]
    final_cols = existing_cols + new_cols

    if final_cols:
        summary_df = summary_df.reindex(columns=final_cols)
        new_row_df = new_row_df.reindex(columns=final_cols)

    return pd.concat([summary_df, new_row_df], ignore_index=True)


def save_case_output(
    df: pd.DataFrame,
    method: str,
    weather: str,
    location: str,
    output_dir: str = "output",
    extra_metrics: dict | None = None,
    summary_filename: str = "summary_results.csv",
    index: bool = False,
) -> tuple[Path, Path]:
    """
    保存单个案例的输出结果。

    功能：
    1) 保存当前方法的时间序列 CSV；
    2) 将当前案例的统计量追加到 summary_results.csv；
    3) summary_results 的顺序严格按 save_case_output 的调用顺序写入，
       不再做任何排序。

    Returns
    -------
    ts_path : Path
        当前案例时间序列 CSV 的路径。
    summary_path : Path
        summary_results.csv 的路径。
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    method_tag = _sanitize_filename(method)
    weather_tag = _sanitize_filename(weather)
    location_tag = _sanitize_filename(location)

    # 1) 保存时间序列
    ts_filename = f"{method_tag}_{weather_tag}_{location_tag}.csv"
    ts_path = output_path / ts_filename
    df.to_csv(ts_path, index=index, encoding="utf-8-sig")

    # 2) 构造当前案例的 summary 行
    summary_row = build_summary_from_df(
        df=df,
        method=method,
        weather=weather,
        location=location,
        extra_metrics=extra_metrics,
    )
    new_row_df = pd.DataFrame([summary_row])

    # 3) 读取旧 summary，并按“运行顺序”追加，不排序
    summary_path = output_path / summary_filename
    if summary_path.exists():
        summary_df = pd.read_csv(summary_path)
        summary_df = _append_summary_row_preserve_runtime_order(
            summary_df=summary_df,
            new_row_df=new_row_df,
            method=method,
            weather=weather,
            location=location,
        )
    else:
        summary_df = new_row_df.copy()

    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    return ts_path, summary_path
