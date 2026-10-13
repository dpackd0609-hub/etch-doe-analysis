"""확인 실험(재현성) 검증 · 공정능력지수(Cpk) 평가 · Robust 범위 점검"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from . import doe2 as d2


@dataclass
class ConfirmResult:
    data: pd.DataFrame
    mean: float
    sd: float
    pred_mean: float
    pi_low: float
    pi_high: float
    all_in_pi: bool
    t_p: float                # H0: 실측 평균 = 모델 예측
    cpk: float
    cpk_lcl: float            # Cpk 95% 하한 (Bissell 근사)
    n_needed: int             # 같은 Cpk·sd 가정 시 하한이 목표를 넘기 위한 최소 표본 수
    figure: Path


def cpk_upper(mean: float, sd: float, usl: float) -> float:
    """Score는 '낮을수록 양호'한 단측 규격 → Cpk = (USL − μ) / 3σ"""
    return (usl - mean) / (3 * sd)


def cpk_lower_bound(cpk: float, n: int, conf: float = 0.95) -> float:
    z = stats.norm.ppf(conf)
    return cpk - z * np.sqrt(1 / (9 * n) + cpk**2 / (2 * (n - 1)))


def required_n(cpk: float, target: float, conf: float = 0.95, n_max: int = 500) -> int:
    for n in range(3, n_max):
        if cpk_lower_bound(cpk, n, conf) >= target:
            return n
    return -1


def run(df: pd.DataFrame, d2res: d2.Doe2Result, cfg: dict, out: Path) -> ConfirmResult:
    c = cfg["confirmation"]
    y = df["Score"].to_numpy()
    mean, sd = y.mean(), y.std(ddof=1)
    p = d2res.opt_pred
    lo, hi = p["obs_ci_lower"], p["obs_ci_upper"]
    t_p = stats.ttest_1samp(y, p["mean"]).pvalue
    cpk = cpk_upper(mean, sd, c["USL"])
    lcl = cpk_lower_bound(cpk, len(y))
    n_need = required_n(cpk, c["target_cpk"])

    fig_path = out / "confirm_runs.png"
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.axhspan(lo, hi, color="tab:green", alpha=0.15, label="95% prediction interval")
    ax.axhline(p["mean"], color="tab:green", ls="--", label=f"model prediction {p['mean']:.2f}")
    ax.axhline(c["USL"], color="tab:red", ls=":", label=f"USL {c['USL']}")
    ax.plot(range(1, len(y) + 1), y, "ko-", label="confirmation runs")
    ax.set_xlabel("Trial"); ax.set_ylabel("Score"); ax.set_xticks(range(1, len(y) + 1))
    ax.set_title("Confirmation runs at optimum recipe")
    ax.legend(fontsize=8, loc="center right"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(fig_path, dpi=150); plt.close(fig)

    return ConfirmResult(df, mean, sd, p["mean"], lo, hi,
                         bool(((y >= lo) & (y <= hi)).all()), t_p, cpk, lcl, n_need, fig_path)


def robust_grid(d2res: d2.Doe2Result, cfg: dict) -> pd.DataFrame:
    """최적점 주변 ±흔들림(3×3)에서의 예측 Score — 넓은 process window 여부 확인."""
    r, o = cfg["robust"], d2res.opt_actual
    rows = []
    for dr in (-r["CHF3_ratio_delta"], 0, r["CHF3_ratio_delta"]):
        for dp in (-r["Pressure_delta"], 0, r["Pressure_delta"]):
            chf3, pr = o["CHF3_ratio"] + dr, o["Pressure"] + dp
            x = d2.to_coded(chf3, pr, cfg)
            pred = float(d2res.model.predict(pd.DataFrame({"x1": [x[0]], "x2": [x[1]]}))[0])
            rows.append({"CHF3_ratio": round(chf3, 4), "Pressure": round(pr, 1),
                         "pred_Score": round(pred, 3)})
    g = pd.DataFrame(rows)
    g["delta_vs_opt"] = (g["pred_Score"] - g["pred_Score"].min()).round(3)
    return g
