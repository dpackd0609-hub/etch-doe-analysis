"""2차 DOE: Gas 조성비(x1) · Pressure(x2) 2인자 CCD 반응표면분석(RSM)

1차 DOE에서 Gas를 범주형(CF4/Mix1/Mix2)으로만 다루면 수준 사이의 최적점을 찾을 수 없으므로,
CHF3 비율이라는 연속 물리량으로 재정의(x1)하고 2차 모형으로 곡률을 추정한다.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

FORMULA = "Score ~ x1 + x2 + x1:x2 + I(x1**2) + I(x2**2)"


@dataclass
class Doe2Result:
    model: object
    anova: pd.DataFrame
    stationary_coded: np.ndarray
    stationary_kind: str
    eigvals: np.ndarray
    opt_coded: np.ndarray
    opt_actual: dict
    opt_pred: pd.Series                     # 예측 평균 · 신뢰구간 · 예측구간
    figures: list[Path] = field(default_factory=list)


# ------------------------------------------------------------------ 코드 변환
def to_actual(coded: np.ndarray, cfg: dict) -> dict:
    c, s = cfg["doe2"]["center"], cfg["doe2"]["step"]
    return {"CHF3_ratio": c["CHF3_ratio"] + coded[0] * s["CHF3_ratio"],
            "Pressure": c["Pressure"] + coded[1] * s["Pressure"]}


def to_coded(chf3: float, pressure: float, cfg: dict) -> np.ndarray:
    c, s = cfg["doe2"]["center"], cfg["doe2"]["step"]
    return np.array([(chf3 - c["CHF3_ratio"]) / s["CHF3_ratio"],
                     (pressure - c["Pressure"]) / s["Pressure"]])


# ------------------------------------------------------------------ 설계
def ccd_design(cfg: dict) -> pd.DataFrame:
    """회전형 중심합성설계: 요인점 4 + 축점 4 + 중심점 n_center."""
    a = cfg["doe2"]["alpha_axial"]
    pts = ([(-1, -1, "factorial"), (1, -1, "factorial"), (-1, 1, "factorial"), (1, 1, "factorial")]
           + [(-a, 0, "axial"), (a, 0, "axial"), (0, -a, "axial"), (0, a, "axial")]
           + [(0, 0, "center")] * cfg["doe2"]["n_center"])
    d = pd.DataFrame(pts, columns=["x1", "x2", "PointType"])
    act = d.apply(lambda r: pd.Series(to_actual(np.array([r.x1, r.x2]), cfg)), axis=1)
    d = pd.concat([d, act], axis=1)
    d["CHF3_ratio"] = d["CHF3_ratio"].round(4)
    d["Pressure"] = d["Pressure"].round(1)
    d["Bias"] = cfg["doe2"]["bias_fixed"]
    d.insert(0, "Run", range(1, len(d) + 1))
    return d


# ------------------------------------------------------------------ 분석
def rsm_anova(df: pd.DataFrame) -> pd.DataFrame:
    """R rsm 패키지와 같은 형식: FO / TWI / PQ 순차 SS + Lack of fit / Pure error."""
    steps = [("FO", "Score ~ x1 + x2"),
             ("TWI", "Score ~ x1 + x2 + x1:x2"),
             ("PQ", FORMULA)]
    rows, prev = [], smf.ols("Score ~ 1", data=df).fit()
    for name, f in steps:
        m = smf.ols(f, data=df).fit()
        rows.append([name, prev.df_resid - m.df_resid, prev.ssr - m.ssr])
        prev = m
    full = prev
    # 순수오차: 동일 조건 반복점 내 변동
    g = df.groupby(["x1", "x2"])["Score"]
    ss_pe = float(((df["Score"] - g.transform("mean")) ** 2).sum())
    df_pe = int((g.count() - 1).sum())
    ss_lof, df_lof = full.ssr - ss_pe, int(full.df_resid) - df_pe
    rows += [["Residuals", full.df_resid, full.ssr],
             ["Lack of fit", df_lof, ss_lof],
             ["Pure error", df_pe, ss_pe]]
    t = pd.DataFrame(rows, columns=["term", "df", "sum_sq"]).set_index("term")
    t["mean_sq"] = t["sum_sq"] / t["df"]
    mse = t.loc["Residuals", "mean_sq"]
    for k in ["FO", "TWI", "PQ"]:
        t.loc[k, "F"] = t.loc[k, "mean_sq"] / mse
        t.loc[k, "p_value"] = stats.f.sf(t.loc[k, "F"], t.loc[k, "df"], full.df_resid)
    if df_pe > 0 and ss_pe > 0:
        t.loc["Lack of fit", "F"] = t.loc["Lack of fit", "mean_sq"] / t.loc["Pure error", "mean_sq"]
        t.loc["Lack of fit", "p_value"] = stats.f.sf(t.loc["Lack of fit", "F"], df_lof, df_pe)
    return t


def canonical(model) -> tuple[np.ndarray, np.ndarray, str]:
    """정준분석: y = b0 + b'x + x'Bx → 정상점 xs = -½B⁻¹b, B의 고유값 부호로 형태 판정."""
    p = model.params
    b = np.array([p["x1"], p["x2"]])
    B = np.array([[p["I(x1 ** 2)"], p["x1:x2"] / 2],
                  [p["x1:x2"] / 2, p["I(x2 ** 2)"]]])
    xs = -0.5 * np.linalg.solve(B, b)
    ev = np.linalg.eigvalsh(B)
    kind = "minimum" if (ev > 0).all() else "maximum" if (ev < 0).all() else "saddle"
    return xs, ev, kind


def constrained_optimum(model, radius: float, n: int = 401) -> np.ndarray:
    """설계영역(코드 반경 ≤ 축점거리) 내부에서 예측 Score 최소점 탐색 — 외삽 방지."""
    g = np.linspace(-radius, radius, n)
    X1, X2 = np.meshgrid(g, g)
    mask = X1**2 + X2**2 <= radius**2 + 1e-9
    pred = model.predict(pd.DataFrame({"x1": X1[mask], "x2": X2[mask]})).to_numpy()
    i = pred.argmin()
    return np.array([X1[mask][i], X2[mask][i]])


def _plot_contour(model, cfg, design: pd.DataFrame, opt: np.ndarray, path: Path):
    a = cfg["doe2"]["alpha_axial"]
    g = np.linspace(-a, a, 200)
    X1, X2 = np.meshgrid(g, g)
    Z = model.predict(pd.DataFrame({"x1": X1.ravel(), "x2": X2.ravel()})).to_numpy().reshape(X1.shape).copy()
    Z[X1**2 + X2**2 > a**2] = np.nan            # 설계영역(원) 밖은 표시하지 않음 — 외삽 방지
    act_x = [to_actual(np.array([v, 0]), cfg)["CHF3_ratio"] for v in g]
    act_y = [to_actual(np.array([0, v]), cfg)["Pressure"] for v in g]
    A1, A2 = np.meshgrid(act_x, act_y)

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    cf = ax.contourf(A1, A2, Z, levels=20, cmap="RdYlGn_r")
    cs = ax.contour(A1, A2, Z, levels=10, colors="k", linewidths=0.5)
    ax.clabel(cs, fontsize=7, fmt="%.2f")
    fig.colorbar(cf, ax=ax, label="predicted Score (lower = better)")
    ax.scatter(design["CHF3_ratio"], design["Pressure"], c="k", s=18, label="CCD runs")
    oa = to_actual(opt, cfg)
    ax.scatter(oa["CHF3_ratio"], oa["Pressure"], marker="*", s=260, c="white",
               edgecolors="k", label="optimum", zorder=5)
    ax.set_xlabel("CHF3 ratio in CF4/CHF3"); ax.set_ylabel("Pressure [mTorr]")
    ax.set_title(f"RSM contour (Bias fixed {cfg['doe2']['bias_fixed']} W)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def run(df: pd.DataFrame, cfg: dict, out: Path) -> Doe2Result:
    model = smf.ols(FORMULA, data=df).fit()
    anova = rsm_anova(df)
    xs, ev, kind = canonical(model)
    radius = cfg["doe2"]["alpha_axial"]
    inside = np.hypot(*xs) <= radius
    opt = xs if (kind == "minimum" and inside) else constrained_optimum(model, radius)
    pred = model.get_prediction(pd.DataFrame({"x1": [opt[0]], "x2": [opt[1]]})) \
                .summary_frame(alpha=0.05).iloc[0]
    fig = out / "doe2_rsm_contour.png"
    _plot_contour(model, cfg, df, opt, fig)
    return Doe2Result(model, anova, xs, kind, ev, opt, to_actual(opt, cfg), pred, [fig])
