"""1차 DOE (3인자 3수준 완전요인) 통계분석: 상관분석 · 주효과도 · ANOVA · 회귀분석 · 인자 선별"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

FACTORS = ["Gas", "Pressure", "Bias"]


@dataclass
class Doe1Result:
    corr: pd.DataFrame
    r_y1_y2: float
    p_y1_y2: float
    anova: dict[str, pd.DataFrame]                 # 응답별 ANOVA 표
    reg_ordinal: object                            # Gas를 1·2·3 코드로 넣은 선형회귀 (기존 방식)
    reg_ratio: object                              # Gas를 CHF3 비율(연속) + 2차항으로 넣은 회귀
    significant: list[str] = field(default_factory=list)
    fixed: dict[str, float] = field(default_factory=dict)
    figures: list[Path] = field(default_factory=list)


# ------------------------------------------------------------------ 분석 함수
def correlation(df: pd.DataFrame, score: str) -> pd.DataFrame:
    cols = ["CHF3_ratio", "Pressure", "Bias", "Tapered_Y1", "Notching_Y2", score]
    return df[cols].corr()


def anova_table(df: pd.DataFrame, response: str) -> pd.DataFrame:
    """범주형 3인자 주효과 ANOVA (Type II). 완전요인·균형 설계이므로 Type I과 동일."""
    model = smf.ols(f"{response} ~ C(Gas) + C(Pressure) + C(Bias)", data=df).fit()
    tbl = sm.stats.anova_lm(model, typ=2)
    tbl["eta_sq"] = tbl["sum_sq"] / tbl["sum_sq"].sum()      # 기여율
    return tbl.rename(columns={"PR(>F)": "p_value"})


def regressions(df: pd.DataFrame, score: str):
    d = df.copy()
    d["Gas_code"] = d["Gas"].cat.codes + 1                  # 1, 2, 3 (등간격 가정)
    ordinal = smf.ols(f"{score} ~ Gas_code + Pressure + Bias", data=d).fit()
    ratio = smf.ols(f"{score} ~ CHF3_ratio + I(CHF3_ratio**2) + Pressure + Bias", data=d).fit()
    return ordinal, ratio


def select_factors(anova_score: pd.DataFrame, alpha: float, cfg: dict):
    """Score ANOVA에서 유의한 인자만 2차 DOE로 넘기고 나머지는 고정."""
    sig, fixed = [], {}
    for f in FACTORS:
        p = anova_score.loc[f"C({f})", "p_value"]
        if p < alpha:
            sig.append(f)
        elif f == "Bias":
            fixed[f] = cfg["doe2"]["bias_fixed"]
    return sig, fixed


# ------------------------------------------------------------------ 시각화
def _plot_heatmap(corr: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr)), corr.index)
    for i in range(len(corr)):
        for j in range(len(corr)):
            v = corr.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color="white" if abs(v) > 0.6 else "black", fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("DOE1 correlation matrix")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def _plot_tradeoff(df: pd.DataFrame, r: float, path: Path):
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    for g, sub in df.groupby("Gas", observed=True):
        ax.scatter(sub["Tapered_Y1"], sub["Notching_Y2"], label=str(g), s=40)
    b1, b0 = np.polyfit(df["Tapered_Y1"], df["Notching_Y2"], 1)
    xs = np.linspace(df["Tapered_Y1"].min(), df["Tapered_Y1"].max(), 50)
    ax.plot(xs, b0 + b1 * xs, "k--", lw=1)
    ax.set_xlabel("Tapered_Y1"); ax.set_ylabel("Notching_Y2")
    ax.set_title(f"Tapered vs Notching trade-off (r = {r:.2f})")
    ax.legend(title="Gas"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def _plot_main_effects(df: pd.DataFrame, responses: list[str], path: Path):
    fig, axes = plt.subplots(len(responses), 3, figsize=(10, 2.8 * len(responses)),
                             sharey="row")
    for i, resp in enumerate(responses):
        grand = df[resp].mean()
        for j, f in enumerate(FACTORS):
            ax = axes[i, j]
            m = df.groupby(f, observed=True)[resp].mean()
            ax.plot([str(x) for x in m.index], m.values, "o-")
            ax.axhline(grand, color="gray", ls=":", lw=1)
            if i == 0:
                ax.set_title(f)
            if j == 0:
                ax.set_ylabel(f"mean {resp}")
            ax.grid(alpha=0.3)
    fig.suptitle("DOE1 main effects")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


# ------------------------------------------------------------------ 실행
def run(df: pd.DataFrame, cfg: dict, out: Path) -> Doe1Result:
    score = cfg["score"]["name"]
    alpha = cfg["doe1"]["alpha"]
    corr = correlation(df, score)
    r, p = stats.pearsonr(df["Tapered_Y1"], df["Notching_Y2"])

    anova = {resp: anova_table(df, resp) for resp in ["Tapered_Y1", "Notching_Y2", score]}
    ordinal, ratio = regressions(df, score)
    sig, fixed = select_factors(anova[score], alpha, cfg)

    figs = [out / "doe1_corr_heatmap.png", out / "doe1_tradeoff.png", out / "doe1_main_effects.png"]
    _plot_heatmap(corr, figs[0])
    _plot_tradeoff(df, r, figs[1])
    _plot_main_effects(df, ["Tapered_Y1", "Notching_Y2", score], figs[2])

    return Doe1Result(corr, r, p, anova, ordinal, ratio, sig, fixed, figs)
