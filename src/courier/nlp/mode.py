"""Công tắc chế độ cho khối 2 (bước 5B). Đặt bằng biến môi trường TRƯỚC khi import courier.nlp:

  COURIER_NLP_MODE        "final" (mặc định) | "blind"
      blind = "chế độ mù": từ điển/luật/mô hình CHỈ xây từ train (+ danh sách động não brainstorm.py).
              Mọi cụm từ trong lexicon.py không xuất hiện trong câu train bị gỡ (vì có thể rút ra từ val-A);
              lưới an toàn chỉ học từ train. Dùng để đo trung thực trên toàn bộ 300 cảnh validation.
      final = cơ chế tốt nhất + học thêm từ toàn bộ validation (từ điển cũ có luật rút từ val-A được bật lại).
  COURIER_NLP_BRAINSTORM  "1" (mặc định) | "0"   dùng danh sách động não
  COURIER_NLP_CONSTRAINTS "1" (mặc định) | "0"   kiểm tra chéo với địa điểm trên bản đồ
  COURIER_NLP_MODEL       "1" (mặc định) | "0"   dùng mô hình ngôn ngữ (nếu đã huấn luyện)
  COURIER_FLAG_V2         "1" (mặc định) | "0"   gia cố gấp/dễ vỡ của bước 5C (0 = như bước 5B)
"""
import os

MODE = os.environ.get("COURIER_NLP_MODE", "final")
assert MODE in ("final", "blind"), MODE
BRAINSTORM = os.environ.get("COURIER_NLP_BRAINSTORM", "1") == "1"
CONSTRAINTS = os.environ.get("COURIER_NLP_CONSTRAINTS", "1") == "1"
MODEL = os.environ.get("COURIER_NLP_MODEL", "1") == "1"
# bước 5C (phần A): động não gấp/dễ vỡ lượt 3 + phủ định có phạm vi + hạn giờ + biểu quyết nhiều nguồn
FLAG_V2 = os.environ.get("COURIER_FLAG_V2", "1") == "1"


def describe():
    return (f"mode={MODE} brainstorm={int(BRAINSTORM)} constraints={int(CONSTRAINTS)} model={int(MODEL)} "
            f"flag_v2={int(FLAG_V2)}")


def train_texts():
    """Câu mission của train (mỗi cảnh 1 câu)."""
    from ..data import load_split
    rows, _, _ = load_split("train")
    return [r["mission"] for r in rows[::10]]
