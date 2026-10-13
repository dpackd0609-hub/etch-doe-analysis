# Dry Etching Defect 개선 — DOE 분석 자동화 파이프라인

Dry Etching 공정에서 **Tapered Sidewall**과 **Bottom Notching**이 trade-off 관계로 함께 나타나는 문제를
실험계획법(DOE)과 반응표면분석(RSM)으로 해결하는 분석 파이프라인입니다.
전처리부터 상관분석, ANOVA, 회귀, RSM 최적화, 확인실험·Cpk, 보고서 작성까지 **명령 한 줄**로 재실행됩니다.

📄 **분석 보고서 전문: [`outputs/report.md`](outputs/report.md)**

> ⚠️ `data/`의 값은 실측이 아닌 **참고용(가상 공정) 데이터**입니다 ([`simulator.py`](src/etchdoe/simulator.py)).
> 실측 SEM 판독값으로 CSV를 교체하면 같은 코드로 그대로 분석됩니다.

---

## 결과 요약

| 구분 | CF4 : CHF3 | Pressure | Bias | 통합 불량 Score ↓ |
|---|---|---|---|---|
| 현 공정 | 100 : 0 | 100 mTorr | 50 W | 9.91 |
| **개선 Recipe** | **75.5 : 24.5** | **44.8 mTorr** | **30 W** | **8.97 (−9.5%)** |

| 단계 | 핵심 결과 |
|---|---|
| 상관분석 | Tapered–Notching r = **−0.98** → 개별 불량이 아닌 통합 Score로 최적화 |
| 1차 ANOVA | Gas · Pressure 유의, **Bias p = 0.93** (개별 불량엔 유의하나 Score에선 상쇄) → 30 W 고정 |
| 변수 재정의 | Gas를 범주 코드(1,2,3) → **CHF3 비율 연속변수 + 2차항**: R² 0.88 → 0.98 |
| 2차 RSM (CCD 13 runs) | R² = **0.983**, x1 1·2차항 모두 유의, Lack-of-fit p = 0.63 |
| 확인실험 (5 runs) | 5회 모두 95% 예측구간 내, 예측 8.97 vs 실측 평균 8.97 |
| 실행 시간 | 전체 파이프라인 약 **1.5초** |

<p align="center">
  <img src="outputs/doe1_tradeoff.png" width="45%">
  <img src="outputs/doe2_rsm_contour.png" width="45%">
</p>
<p align="center">
  <img src="outputs/doe1_main_effects.png" width="70%">
</p>

---

## 문제 정의

- **현상**: 현 공정(CF4 단독 / 100 mTorr / 50 W)에서 Positive Taper와 Bottom Notching이 동시에 발생
- **원인 가설**: 라디칼(화학 식각) · 이온(방향성 식각) · CFx 폴리머(측벽 패시베이션)의 균형 붕괴
  - CHF3 ↑ → 폴리머 패시베이션 ↑ → Taper ↑, Notching ↓
  - Bias ↑ → 이온 에너지 ↑ → Taper ↓, 하부 corner 공격 ↑ → Notching ↑
  - Pressure ↓ → 이온–중성 충돌 ↓ → 수직 직진성 ↑
- **핵심 인자**: Gas 조성(CF4:CHF3), Pressure, Bias Power

## 분석 흐름

```
1차 DOE  Gas(CF4 / Mix1 8:2 / Mix2 7:3) × Pressure(70/100/130) × Bias(30/50/70) = 3³ = 27 runs
   │  상관분석  → trade-off 확인 → 통합 Score = Tapered + Notching
   │  ANOVA     → 유의 인자 자동 선별, 비유의 인자(Bias) 고정
   │  회귀분석  → Gas를 범주가 아닌 CHF3 비율(연속 물리량)로 재정의
   ▼
2차 DOE  CHF3 비율(x1) × Pressure(x2) 회전형 CCD = 13 runs (중심점 5회 반복)
   │  2차 모형 · RSM ANOVA(FO/TWI/PQ/Lack-of-fit) · 정준분석
   │  정상점이 설계영역 밖이면 외삽 대신 영역 내 최소점 채택
   ▼
확인실험 5 runs → 95% 예측구간 검증 · Cpk 및 신뢰하한 · Robust(±흔들림) 점검
```

## AI 활용 방식

| 역할 | 담당 |
|---|---|
| 반복·정형 작업 (전처리, 검정 코드, 시각화, 보고서 생성) | AI (Claude Code) |
| 인자 선정, 통계 결과 타당성 검증, 실험 구조 재설계 | 본인 |

- 처음 생성된 코드는 Gas를 **범주형 3수준**으로만 다뤄 수준 사이의 최적점을 찾을 수 없었음
- CF4 / Mix1 / Mix2가 **CHF3 비율이라는 하나의 연속 물리량 위의 점**이라는 점과 주효과도의 단조 경향을 근거로
  연속변수 x1로 재정의하고 RSM(CCD)으로 설계를 변경
- 함수화와 설정 파일 분리로 **데이터와 인자만 바꾸면 재실행** — 1차 분석 코드를 2차 DOE에 그대로 재사용

## 실행 방법

```bash
git clone <this-repo>
cd etch-doe-analysis
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python run_pipeline.py             # data/*.csv로 분석 → outputs/ 갱신
python run_pipeline.py --regen     # 참고용 데이터 새로 생성 후 분석
pytest -q                          # 테스트
```

VS Code에서는 **F5 → "분석 파이프라인 실행"**, 테스트는 Testing 탭에서 실행할 수 있습니다.

## 폴더 구조

```
etch-doe-analysis/
├── run_pipeline.py          # 실행 진입점
├── config.yaml              # 인자 수준 · 데이터 경로 · 유의수준 · 규격(USL) 등 모든 설정
├── data/                    # doe1_raw.csv (27) · doe2_raw.csv (13) · confirm_raw.csv (5)
├── src/etchdoe/
│   ├── preprocess.py        # Gas 라벨 → CHF3 비율 매핑, 이상치 점검, Score 계산
│   ├── doe1.py              # 상관분석 · 주효과도 · ANOVA · 회귀 · 인자 선별
│   ├── doe2.py              # CCD 설계 · RSM 적합 · RSM ANOVA · 정준분석 · 등고선
│   ├── confirm.py           # 확인실험 검증 · Cpk(+신뢰하한) · Robust 점검
│   ├── report.py            # Markdown 보고서 생성
│   ├── simulator.py         # 참고용 가상 공정 (실측 데이터가 있으면 불필요)
│   └── pipeline.py          # 전체 흐름 연결
├── tests/test_pipeline.py   # pytest (정준분석 정확도, CCD 구조, Cpk, end-to-end)
└── outputs/                 # report.md + 그래프
```

## 실측 데이터로 교체하기

| 파일 | 필수 컬럼 |
|---|---|
| `doe1_raw.csv` | `Gas`(CF4/Mix1/Mix2), `Pressure`, `Bias`, `Tapered_Y1`, `Notching_Y2` |
| `doe2_raw.csv` | `x1`, `x2`, `CHF3_ratio`, `Pressure`, `Bias`, `Tapered_Y1`, `Notching_Y2` |
| `confirm_raw.csv` | `Trial`, `CHF3_ratio`, `Pressure`, `Bias`, `Tapered_Y1`, `Notching_Y2` |

응답값은 SEM 판독 기반 0~10 등급(낮을수록 양호)입니다. 가스 조성 수준, CCD 중심·간격, Score 가중치, USL은 `config.yaml`에서 바꿉니다.

## 한계와 다음 단계

- **공정능력**: 가상 데이터의 산포는 SEM 판독 노이즈만 반영합니다. 따라서 Cpk는 참고값이며, 실측 Wafer-to-Wafer · Lot-to-Lot 데이터로 Cpk ≥ 1.33(4σ)을 검증해야 합니다.
- **최적점 위치**: 정상점이 저압 쪽 설계영역 밖에 있어, 영역 경계 내 최소점을 채택했습니다. 저압 방향 추가 실험(최대경사법)이 필요합니다.
- **평가 항목 확장**: Etch rate · Selectivity · CD uniformity를 함께 보는 다중응답 최적화로 넓힐 수 있습니다.

## Tech Stack

Python · pandas · NumPy · SciPy · statsmodels · matplotlib · pytest
