"""pytest: 파이프라인 핵심 로직 검증 (VS Code Testing 탭에서 실행 가능)"""
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from etchdoe import confirm, doe2, pipeline  # noqa: E402


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    """원본 폴더를 건드리지 않도록 임시 폴더에 config를 복사해 전체 파이프라인 실행"""
    tmp = tmp_path_factory.mktemp("run")
    shutil.copy(ROOT / "config.yaml", tmp / "config.yaml")
    return pipeline.run(tmp / "config.yaml", regen=True, verbose=False)


def test_doe1_design_is_full_factorial():
    cfg = pipeline.load_config(ROOT / "config.yaml")
    d = pipeline.doe1_design(cfg)
    assert len(d) == 27 and not d.duplicated(["Gas", "Pressure", "Bias"]).any()


def test_ccd_structure():
    cfg = pipeline.load_config(ROOT / "config.yaml")
    d = doe2.ccd_design(cfg)
    assert d["PointType"].value_counts().to_dict() == {"factorial": 4, "axial": 4, "center": cfg["doe2"]["n_center"]}


def test_canonical_recovers_known_minimum():
    """알려진 2차 함수에서 정상점을 정확히 복원하는지 확인"""
    g = np.linspace(-1.5, 1.5, 7)
    x1, x2 = np.meshgrid(g, g)
    df = pd.DataFrame({"x1": x1.ravel(), "x2": x2.ravel()})
    df["Score"] = 5 + (df.x1 - 0.3) ** 2 + 2 * (df.x2 + 0.5) ** 2
    m = smf.ols(doe2.FORMULA, data=df).fit()
    xs, ev, kind = doe2.canonical(m)
    assert kind == "minimum"
    np.testing.assert_allclose(xs, [0.3, -0.5], atol=1e-8)


def test_cpk():
    assert confirm.cpk_upper(9.0, 0.1, 9.5) == pytest.approx(5 / 3)
    assert confirm.cpk_lower_bound(1.5, 5) < 1.5


def test_pipeline_end_to_end(result):
    r1, r2, r3 = result["doe1"], result["doe2"], result["confirm"]
    assert r1.r_y1_y2 < -0.8                     # trade-off 검출
    assert "Bias" not in r1.significant          # Bias 상쇄 → 비유의
    assert r2.model.rsquared > 0.9
    assert r3.all_in_pi                          # 확인실험이 예측구간 안
    assert result["report"].exists()
