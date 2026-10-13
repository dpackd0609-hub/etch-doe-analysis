"""실행 진입점:  python run_pipeline.py [--config config.yaml] [--regen]"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from etchdoe import pipeline  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Dry Etching defect DOE 분석 파이프라인")
    ap.add_argument("--config", default=str(Path(__file__).resolve().parent / "config.yaml"))
    ap.add_argument("--regen", action="store_true", help="참고용 데이터를 새로 생성 (기존 CSV 덮어씀)")
    args = ap.parse_args()
    pipeline.run(args.config, regen=args.regen)
