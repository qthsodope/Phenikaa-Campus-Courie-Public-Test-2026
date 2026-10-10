"""MỘT LỆNH: thư mục dữ liệu (observations.json + images/) -> predictions.json. Chạy offline, không gọi dịch vụ ngoài.

    .venv/bin/python scripts/predict.py --data delivery_public/test
    .venv/bin/python scripts/predict.py --data /duong/dan/private_test --out ket_qua.json

Mặc định ghi vào submissions/predictions_<tên thư mục>_<ngày giờ>.json. Dùng được ngay cho bộ dữ liệu mới
(chỉ cần observations.json và thư mục ảnh; không cần scenes.json hay labels.json).
"""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from courier.pipeline import predict_rows, load_rows  # noqa: E402
from courier.vision.reader import VisionReader  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True, help="thư mục chứa observations.json và images/")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--models", type=Path, default=ROOT / "models")
    a = ap.parse_args()

    rows = load_rows(a.data)
    t0 = time.time()
    reader = VisionReader(a.models)
    preds = predict_rows(a.data, rows, reader)
    dt = time.time() - t0

    assert len(preds) == len(rows) and all(isinstance(p, int) and 0 <= p <= 3 for p in preds)
    out = a.out or ROOT / "submissions" / f"predictions_{a.data.name}_{datetime.now():%Y%m%d_%H%M}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(preds), encoding="utf-8")
    print(f"đã ghi {len(preds)} dự đoán ({len(rows) // 10} cảnh) vào {out}")
    print(f"thời gian: {dt:.0f}s ({dt / max(len(rows) // 10, 1):.2f}s/cảnh)")


if __name__ == "__main__":
    main()
