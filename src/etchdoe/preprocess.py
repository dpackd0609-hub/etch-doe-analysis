"""raw data → 분석용 데이터 전처리 (Gas 라벨 → CHF3 비율, 통합 불량 Score 계산)"""
from __future__ import annotations

import pandas as pd

REQUIRED_RESP = ("Tapered_Y1", "Notching_Y2")


def add_score(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """가중합 통합 불량 Score 계산 (낮을수록 양호)."""
    missing = [c for c in REQUIRED_RESP if c not in df.columns]
    if missing:
        raise ValueError(f"응답변수 컬럼이 없습니다: {missing}")
    w = cfg["score"]["weights"]
    out = df.copy()
    out[cfg["score"]["name"]] = sum(out[col] * wt for col, wt in w.items()).round(3)
    return out


def preprocess_doe1(raw: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """1차 DOE raw data 정리.

    - Gas 라벨(CF4/Mix1/Mix2)을 CHF3 비율(연속 물리량)로 매핑해 CHF3_ratio 컬럼 추가
    - 결측·범위 이상치 점검
    - Score 계산
    """
    gas_map = cfg["doe1"]["factors"]["Gas"]["levels"]
    df = raw.copy()
    df["Gas"] = df["Gas"].astype(str).str.strip()
    unknown = set(df["Gas"]) - set(gas_map)
    if unknown:
        raise ValueError(f"config에 정의되지 않은 Gas 라벨: {unknown}")
    df["CHF3_ratio"] = df["Gas"].map(gas_map)

    if df[list(REQUIRED_RESP)].isna().any().any():
        raise ValueError("응답변수에 결측값이 있습니다.")
    if ((df[list(REQUIRED_RESP)] < 0) | (df[list(REQUIRED_RESP)] > 10)).any().any():
        raise ValueError("응답변수가 0~10 등급 범위를 벗어났습니다.")

    # 범주 순서 고정 (CHF3 비율 순)
    order = sorted(gas_map, key=gas_map.get)
    df["Gas"] = pd.Categorical(df["Gas"], categories=order, ordered=True)
    return add_score(df, cfg)
