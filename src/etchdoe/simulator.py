"""
가상 Dry Etching 공정 시뮬레이터 (참고용 raw data 생성기)

실측 SEM 데이터가 없는 상황에서 분석 파이프라인을 검증하기 위한 '참고용' 데이터를 만듭니다.
물리적 경향만 반영한 단순 모델이며, 실제 공정값이 아닙니다.
실측 데이터가 생기면 data/*.csv 를 교체하고 이 모듈은 쓰지 않으면 됩니다.

반영한 경향 (라디칼 · 이온 · CFx 폴리머 균형)
  - CHF3 비율 ↑  → CFx 폴리머 패시베이션 ↑ → Tapered ↑, Notching ↓   (trade-off)
  - Bias ↑       → 이온 에너지 ↑ → 바닥 식각 ↑ → Tapered ↓, 하부 corner 공격 ↑ → Notching ↑
  - Pressure ↑   → 이온-중성 충돌 ↑, 방향성 ↓ → Tapered ↑ (저압일수록 수직 직진성 ↑)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# 시뮬레이터 내부 계수 (분석 코드는 이 값을 모른다고 가정)
#   Score = Tapered + Notching 은 CHF3 비율·압력에 대해 완만한 볼록(convex) 형태 →
#   개별 불량은 크게 움직이지만(trade-off) 합은 내부 최적점에서만 소폭 개선되는 현실적 구조
_P = dict(
    t0=1.0, t_lin=16.0, t_quad=6.0,        # Tapered: CHF3 비율 (패시베이션 ↑ → 기울어짐 ↑)
    n0=8.6, n_lin=-21.0, n_quad=4.0,       # Notching: CHF3 비율 (보호막 ↑ → 파임 ↓)
    p_ref=50.0, p_lin=0.012,               # 압력: Tapered ↑ / Notching ↓ (선형 상쇄)
    p_quad=0.00015,                        # 압력: 저압(≈50 mTorr)에서 방향성 최적 → Tapered 곡률
    b_lin=0.030,                           # Bias: Tapered ↓ / Notching ↑ (서로 상쇄)
    noise=0.03,                            # 측정(SEM 판독) 노이즈 σ
)


def true_response(chf3: np.ndarray, pressure: np.ndarray, bias: np.ndarray):
    chf3, pressure, bias = (np.asarray(x, dtype=float) for x in (chf3, pressure, bias))
    dp, db = pressure - _P["p_ref"], bias - 30
    tap = (_P["t0"] + _P["t_lin"] * chf3 + _P["t_quad"] * chf3**2
           + _P["p_lin"] * dp + _P["p_quad"] * dp**2 - _P["b_lin"] * db)
    notch = (_P["n0"] + _P["n_lin"] * chf3 + _P["n_quad"] * chf3**2
             - _P["p_lin"] * dp + _P["b_lin"] * db)
    return np.clip(tap, 0, 10), np.clip(notch, 0, 10)


def measure(df: pd.DataFrame, rng: np.random.Generator,
            chf3_col="CHF3_ratio", p_col="Pressure", b_col="Bias") -> pd.DataFrame:
    """조건 표(df)에 가상 SEM 판독값(Tapered_Y1, Notching_Y2)을 붙여 반환."""
    tap, notch = true_response(df[chf3_col], df[p_col], df[b_col])
    out = df.copy()
    out["Tapered_Y1"] = np.round(np.clip(tap + rng.normal(0, _P["noise"], len(df)), 0, 10), 2)
    out["Notching_Y2"] = np.round(np.clip(notch + rng.normal(0, _P["noise"], len(df)), 0, 10), 2)
    return out
