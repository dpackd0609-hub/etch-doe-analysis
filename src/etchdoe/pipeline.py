"""전체 분석 파이프라인: 데이터 확보 → 전처리 → 1차 DOE → 인자 선별 → 2차 RSM → 확인실험 → 보고서"""
from __future__ import annotations

import itertools
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import confirm, doe1, doe2, report, simulator
from .preprocess import add_score, preprocess_doe1


def load_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_root"] = Path(path).resolve().parent
    return cfg


def _path(cfg, rel) -> Path:
    return cfg["_root"] / rel


def _get_or_simulate(csv: Path, design: pd.DataFrame, rng, regen: bool, log) -> pd.DataFrame:
    """실측 CSV가 있으면 읽고, 없으면(또는 --regen) 가상 공정으로 '실험'을 수행해 저장."""
    if csv.exists() and not regen:
        log(f"  · 데이터 로드: {csv.name}")
        return pd.read_csv(csv)
    df = simulator.measure(design, rng)
    csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv, index=False, encoding="utf-8-sig")
    log(f"  · 참고용 데이터 생성: {csv.name} ({len(df)} runs)")
    return df


def doe1_design(cfg: dict) -> pd.DataFrame:
    f = cfg["doe1"]["factors"]
    gas_map = f["Gas"]["levels"]
    rows = list(itertools.product(gas_map, f["Pressure"]["levels"], f["Bias"]["levels"]))
    d = pd.DataFrame(rows, columns=["Gas", "Pressure", "Bias"])
    d["CHF3_ratio"] = d["Gas"].map(gas_map)
    d.insert(0, "No", range(1, len(d) + 1))
    return d


def run(config_path: str | Path = "config.yaml", regen: bool = False, verbose: bool = True):
    log = print if verbose else (lambda *a, **k: None)
    t0 = time.perf_counter()
    cfg = load_config(config_path)
    out = _path(cfg, cfg["output_dir"]); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg["seed"])

    # ---------- 1차 DOE
    log("[1/4] 1차 DOE (Gas × Pressure × Bias, 3³ = 27 runs)")
    design1 = doe1_design(cfg)
    raw1 = _get_or_simulate(_path(cfg, cfg["doe1"]["data"]), design1, rng, regen, log)
    df1 = preprocess_doe1(raw1.drop(columns=["CHF3_ratio"], errors="ignore"), cfg)
    r1 = doe1.run(df1, cfg, out)
    log(f"  · Tapered–Notching r = {r1.r_y1_y2:.3f}, 유의 인자 = {r1.significant}, 고정 = {r1.fixed}")

    # ---------- 2차 DOE (RSM)
    log("[2/4] 2차 DOE (CHF3 비율 × Pressure CCD, Bias 고정)")
    design2 = doe2.ccd_design(cfg)
    raw2 = _get_or_simulate(_path(cfg, cfg["doe2"]["data"]), design2, rng, regen, log)
    df2 = add_score(raw2, cfg)
    r2 = doe2.run(df2, cfg, out)
    log(f"  · R² = {r2.model.rsquared:.4f}, 정상점 = {r2.stationary_kind}, "
        f"최적 = CHF3 {r2.opt_actual['CHF3_ratio']:.3f}, P {r2.opt_actual['Pressure']:.1f} mTorr")

    # ---------- 확인 실험
    log("[3/4] 최적 조건 확인 실험")
    n = cfg["confirmation"]["n_runs"]
    design3 = pd.DataFrame({"Trial": range(1, n + 1),
                            "CHF3_ratio": round(r2.opt_actual["CHF3_ratio"], 4),
                            "Pressure": round(r2.opt_actual["Pressure"], 1),
                            "Bias": cfg["doe2"]["bias_fixed"]})
    raw3 = _get_or_simulate(_path(cfg, cfg["confirmation"]["data"]), design3, rng, regen, log)
    df3 = add_score(raw3, cfg)
    r3 = confirm.run(df3, r2, cfg, out)
    robust = confirm.robust_grid(r2, cfg)
    log(f"  · 평균 {r3.mean:.3f} (예측 {r3.pred_mean:.3f}), Cpk {r3.cpk:.2f} (95% 하한 {r3.cpk_lcl:.2f})")

    # ---------- 보고서
    log("[4/4] 보고서 작성")
    base = df1[(df1["CHF3_ratio"] == cfg["baseline"]["CHF3_ratio"])
               & (df1["Pressure"] == cfg["baseline"]["Pressure"])
               & (df1["Bias"] == cfg["baseline"]["Bias"])]
    elapsed = time.perf_counter() - t0
    path = report.write(out, cfg, df1, r1, df2, r2, r3, robust, base, elapsed)
    log(f"완료 ({elapsed:.1f} s) → {path}")
    return dict(cfg=cfg, doe1=r1, doe2=r2, confirm=r3, robust=robust, report=path)
