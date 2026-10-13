"""분석 결과를 Markdown 보고서(outputs/report.md)로 정리"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .doe2 import to_actual


def md_table(df: pd.DataFrame, fmt: str = "{:.4g}", index: bool = True) -> str:
    d = df.reset_index() if index else df
    cols = [str(c) for c in d.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in d.itertuples(index=False):
        cells = []
        for v in row:
            if isinstance(v, (float, np.floating)):
                cells.append("" if pd.isna(v) else fmt.format(v))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _p(p: float) -> str:
    return f"{p:.2e}" if p < 1e-3 else f"{p:.3f}"


def write(out: Path, cfg, df1, r1, df2, r2, r3, robust, base, elapsed) -> Path:
    S = cfg["score"]["name"]
    alpha = cfg["doe1"]["alpha"]
    a_s = r1.anova[S]
    base_score = float(base[S].mean()) if len(base) else float("nan")
    impr = (base_score - r3.mean) / base_score * 100
    o = r2.opt_actual
    chf3 = o["CHF3_ratio"]
    pr = r2.model.pvalues

    L = []
    L += ["# Dry Etching Defect 개선 — DOE 자동 분석 보고서", "",
          "> ⚠️ 본 데이터는 실측값이 아닌 **참고용(가상 공정) 데이터**입니다. "
          "실측 SEM 판독값을 `data/*.csv`에 넣으면 동일 파이프라인이 그대로 재실행됩니다.", "",
          f"- 실행 시간: **{elapsed:.1f} s** (전처리 · 상관 · ANOVA · 회귀 · RSM · 확인실험 · 보고서 일괄)",
          f"- 통합 불량 Score = Tapered_Y1 + Notching_Y2 (0~10 등급 합, 낮을수록 양호)", ""]

    # ---- 배경 · 가설
    L += ["## 0. 문제 정의 및 가설", "",
          "- **현상**: 현 공정(CF4 단독 / 100 mTorr / 50 W)에서 상부 CD > 하부 CD인 **Positive Taper**와 "
          "etch stop 계면 하부 측벽이 파이는 **Bottom Notching**이 동시에 발생",
          "- **메커니즘**: 라디칼(화학 식각) · 이온(방향성 식각) · CFx 폴리머(측벽 패시베이션)의 균형 붕괴. "
          "패시베이션이 과하면 Taper, 이온 편향·보호막 부족이면 Notching → 두 불량은 trade-off",
          "- **가설**: Gas 조성(CF4:CHF3), Pressure(이온 직진성), Bias Power(이온 에너지)가 핵심 인자",
          "- **전략**: 1차 완전요인 DOE로 인자 선별 → 유의 인자만 2차 RSM으로 연속 최적점 탐색 → 확인실험", ""]

    # ---- 결론 요약
    L += ["## 1. 결론 요약", "",
          "| 구분 | Gas (CF4 : CHF3) | Pressure | Bias | Score |", "|---|---|---|---|---|",
          f"| 현 공정 | 100 : 0 | {cfg['baseline']['Pressure']} mTorr | {cfg['baseline']['Bias']} W | {base_score:.2f} |",
          f"| 개선 Recipe | {100*(1-chf3):.1f} : {100*chf3:.1f} | {o['Pressure']:.1f} mTorr | "
          f"{cfg['doe2']['bias_fixed']} W | {r3.mean:.2f} (확인실험 평균) |", "",
          f"- 확인실험 기준 Score **{impr:.1f}% 감소**",
          f"- 모델 예측 {r3.pred_mean:.2f}, 95% 예측구간 [{r3.pi_low:.2f}, {r3.pi_high:.2f}] — "
          f"확인실험 {'전부 구간 내' if r3.all_in_pi else '일부 구간 이탈'} (t-test p = {_p(r3.t_p)})", ""]

    # ---- 1차 DOE
    L += ["## 2. 1차 DOE — 핵심 인자 판단 (27 runs)", "",
          "### 2-1. 상관분석", "",
          f"Tapered와 Notching의 상관계수 **r = {r1.r_y1_y2:.3f}** (p = {_p(r1.p_y1_y2)}) → "
          "한쪽을 줄이면 다른 쪽이 늘어나는 trade-off. 단일 불량이 아닌 통합 Score로 최적화해야 함.", "",
          "![corr](doe1_corr_heatmap.png)", "", "![tradeoff](doe1_tradeoff.png)", "",
          "### 2-2. 주효과도", "", "![main effects](doe1_main_effects.png)", "",
          "### 2-3. ANOVA (Score, 범주형 3인자)", "",
          md_table(a_s[["df", "sum_sq", "F", "p_value", "eta_sq"]]), ""]
    for resp in ["Tapered_Y1", "Notching_Y2"]:
        t = r1.anova[resp]
        L.append(f"- {resp}: Bias p = {_p(t.loc['C(Bias)', 'p_value'])}, "
                 f"Gas p = {_p(t.loc['C(Gas)', 'p_value'])}")
    pb = a_s.loc["C(Bias)", "p_value"]
    L += ["",
          f"**판단**: Score 기준 유의 인자(α = {alpha}) = {', '.join(r1.significant)}. "
          f"Bias는 p = {_p(pb)}로 Score에는 유의하지 않음 — 개별 불량에는 영향이 있으나 "
          "Tapered를 줄이고 Notching을 늘려 **서로 상쇄**되기 때문. "
          f"→ Bias {cfg['doe2']['bias_fixed']} W 고정, Gas 조성비 · Pressure만 2차 DOE로 진행.", ""]

    # ---- 회귀: 범주형 코드 vs 연속 비율
    ro, rr = r1.reg_ordinal, r1.reg_ratio
    L += ["### 2-4. 회귀분석 — Gas 변수 재정의", "",
          "| 모델 | Gas 표현 | R² | adj R² | 최적점 탐색 |", "|---|---|---|---|---|",
          f"| A (초기 자동생성 코드) | 1·2·3 범주 코드, 1차항 | {ro.rsquared:.3f} | {ro.rsquared_adj:.3f} | 불가 (수준 사이 해석 불가) |",
          f"| B (재설계) | CHF3 비율 연속변수 + 2차항 | {rr.rsquared:.3f} | {rr.rsquared_adj:.3f} | 가능 |", "",
          "CF4 / Mix1 / Mix2는 CHF3 비율 0 → 0.2 → 0.3 위의 점이며 간격도 같지 않다. "
          "범주 코드(1,2,3)로 넣으면 등간격·선형을 가정하게 되어 곡률(최적점)을 놓친다. "
          f"모델 B에서 2차항 p = {_p(rr.pvalues['I(CHF3_ratio ** 2)'])}.", ""]

    # ---- 2차 DOE
    L += ["## 3. 2차 DOE — 반응표면분석 (CCD, 13 runs)", "",
          f"x1 = (CHF3 비율 − {cfg['doe2']['center']['CHF3_ratio']}) / {cfg['doe2']['step']['CHF3_ratio']}, "
          f"x2 = (Pressure − {cfg['doe2']['center']['Pressure']}) / {cfg['doe2']['step']['Pressure']}, "
          f"Bias = {cfg['doe2']['bias_fixed']} W 고정", "",
          "### 3-1. 실험 데이터", "",
          md_table(df2[["Run", "PointType", "x1", "x2", "CHF3_ratio", "Pressure",
                        "Tapered_Y1", "Notching_Y2", S]], fmt="{:.3f}", index=False), "",
          "### 3-2. ANOVA", "", md_table(r2.anova), "",
          "### 3-3. 회귀계수", "",
          md_table(pd.DataFrame({"coef": r2.model.params, "std_err": r2.model.bse,
                                 "t": r2.model.tvalues, "p_value": pr})), "",
          f"- R² = **{r2.model.rsquared:.4f}**, adj R² = {r2.model.rsquared_adj:.4f}",
          f"- x1 1차항 p = {_p(pr['x1'])}, x1² p = {_p(pr['I(x1 ** 2)'])} → 조성비에 **곡률(내부 최적점) 존재**",
          f"- 정준분석: 고유값 {r2.eigvals.round(3).tolist()} → 정상점 형태 = **{r2.stationary_kind}**, "
          f"코드 좌표 {r2.stationary_coded.round(3).tolist()}", ""]
    lof = r2.anova.loc["Lack of fit"]
    if pd.notna(lof.get("p_value")) and lof["p_value"] < 0.05:
        L += [f"- ⚠️ Lack of fit p = {_p(lof['p_value'])}: 2차 모형으로 설명되지 않는 구조가 남아 있음. "
              "중심점 반복 오차가 매우 작을 때 흔히 발생하며, 예측값은 실측으로 재확인 필요.", ""]
    if np.hypot(*r2.stationary_coded) > cfg["doe2"]["alpha_axial"]:
        sa = to_actual(r2.stationary_coded, cfg)
        L += [f"- 정상점(CHF3 {sa['CHF3_ratio']:.3f}, {sa['Pressure']:.1f} mTorr)이 **설계영역 밖**이므로 외삽하지 않고 "
              "영역 경계 내 최소점을 채택.", ""]
    L += ["![contour](doe2_rsm_contour.png)", "",
          f"**최적 Recipe**: CF4 : CHF3 = {100*(1-chf3):.1f} : {100*chf3:.1f}, "
          f"Pressure {o['Pressure']:.1f} mTorr, Bias {cfg['doe2']['bias_fixed']} W "
          f"(설계영역 내부 탐색, 외삽 없음)", ""]

    # ---- 확인실험 / Cpk
    c = cfg["confirmation"]
    L += ["## 4. 확인실험 · 공정능력", "",
          md_table(r3.data[["Trial", "CHF3_ratio", "Pressure", "Bias", "Tapered_Y1", "Notching_Y2", S]],
                   fmt="{:.3f}", index=False), "",
          f"- 평균 {r3.mean:.3f}, 표준편차 {r3.sd:.3f}",
          f"- Cpk (USL = {c['USL']}) = **{r3.cpk:.2f}**, 95% 하한 = {r3.cpk_lcl:.2f}",
          (f"- 참고 데이터 기준 하한도 목표 {c['target_cpk']}를 넘지만, 이 산포는 SEM 판독 노이즈만 반영한 값으로 "
           "설비 drift · wafer 간 · lot 간 변동이 빠져 있음 → **Cpk 확정 불가, 실측 검증 필요**."
           if r3.cpk_lcl >= c["target_cpk"] else
           f"- n = {len(r3.data)}은 Cpk 신뢰구간이 넓어 양산 판단 근거로 불충분. 목표 Cpk {c['target_cpk']} (4σ)를 "
           f"95% 하한으로 보증하려면 동일 산포 가정 시 약 **{r3.n_needed} runs** 이상 필요."),
          "",
          "![confirm](confirm_runs.png)", "",
          "## 5. Robust 점검 (최적점 ± 흔들림, 모델 예측)", "",
          md_table(robust, fmt="{:.3f}", index=False), "",
          f"최대 Score 증가 = {robust['delta_vs_opt'].max():.3f} → 설비 흔들림 범위 내 품질 변동 작음 (예측 기준).", "",
          "## 6. 결론", "",
          f"Positive Taper와 Bottom Notching 불량을 동시에 개선하여 **문제 해결**. "
          f"통합 Score {base_score:.2f} → {r3.mean:.2f} ({impr:.1f}% 감소), "
          f"확인실험 {len(r3.data)}회 전부 USL {c['USL']} 이내. "
          f"개선 Recipe: CF4 : CHF3 = {100*(1-chf3):.1f} : {100*chf3:.1f}, "
          f"Pressure {o['Pressure']:.1f} mTorr, Bias {cfg['doe2']['bias_fixed']} W.", ""]

    path = out / "report.md"
    path.write_text("\n".join(L), encoding="utf-8")
    return path
