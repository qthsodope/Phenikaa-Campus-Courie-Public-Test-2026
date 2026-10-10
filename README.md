# Phenikaa Campus Courier — The Codeaholics (vòng Public Test)

Dự đoán bước đi đầu tiên (UP/DOWN/LEFT/RIGHT) của 10 robot giao hàng từ ảnh sơ đồ campus + yêu cầu tiếng Việt
(vòng public test "Campus Courier 2026").

Thành viên: 23010680 – Nguyễn Quốc Thiên · 23010174 – Đỗ Thành Long · 23010069 – Trịnh Gia Huy.

## 1. Cài đặt

Đã kiểm tra trên: Ubuntu, Python 3.14, PyTorch 2.14.1 + CUDA 13.0, GPU RTX 3060 Laptop 6 GB.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/download_models.py     # MỘT LẦN, cần mạng: PhoBERT-base gốc (vinai, giấy phép MIT)
```

Đặt dữ liệu của BTC ở `delivery_public/` (`train/ validation/ test/ robots.json`). Lúc dự đoán, thư mục
`delivery_public/train` và `validation` phải có mặt: lưới an toàn của khối câu tự học lại từ câu train + validation
(khoảng 20 giây), không đọc gì của tập cần đoán ngoài `observations.json` và ảnh.

## 2. Trọng số mô hình

| file | là gì | tham số | trạng thái |
|---|---|---|---|
| `models/kp.pt` `edge.pt` `node.pt` `misc.pt` `config.json` | khối ảnh: dò điểm (ResNet18) + 3 CNN nhỏ | 5,08 triệu | có trong repo, đúng bản đã dùng để nộp |
| `models/nlp_final.pt` | PhoBERT-base tinh chỉnh, 8 đầu ra (khối câu) | 134,5 triệu | **không còn** (xem dưới) |

**Vì sao không còn `nlp_final.pt`:** sau vòng public test, nhóm chuyển sang vòng private test và huấn luyện lại khối
câu cho bộ dữ liệu mới. Do sơ suất, script huấn luyện ghi file mới vào đúng đường dẫn `models/nlp_final.pt`, nên file
của vòng public test bị ghi đè; nhóm không giữ bản sao riêng. Code, dữ liệu huấn luyện và cấu hình vẫn còn nguyên nên
có thể học lại (mục 4); mô hình học lại sẽ rất gần nhưng không giống từng byte bản cũ (GPU không tất định tuyệt đối),
nên file dự đoán có thể lệch vài dòng so với bài đã nộp.

Thiếu `nlp_final.pt` thì `predict.py` vẫn chạy, khi đó khối câu chỉ dùng luật + từ điển + lưới an toàn (kém hơn).

## 3. Dự đoán (một lệnh, offline)

```bash
.venv/bin/python scripts/predict.py --data delivery_public/test --out predictions.json
```
1.200 cảnh khoảng 4 phút trên RTX 3060; kết quả tất định.

`submissions/BEST.json` là file dự đoán nhóm đã nộp ở vòng public test (sha256
`cb2521791000af083474a4db9aa804224a1277eb2e4b1f3680b5c6a41b54c43d`). Vì thiếu `nlp_final.pt` gốc (mục 2), chạy lại
`predict.py` có thể cho kết quả lệch so với file này.

## 4. Huấn luyện lại từ đầu (seed 2026)

```bash
# khối câu: PhoBERT chế độ cuối (train + toàn bộ validation), ~38 phút GPU -> models/nlp_final.pt
COURIER_NLP_MODE=final .venv/bin/python scripts/train_nlp_model.py --backbone phobert
# khối ảnh (bản đã nộp, "C2"): dò điểm học tiếp từ bản bước 4, rồi 3 mạng mảnh ảnh; ~10 + ~23 phút
.venv/bin/python scripts/train_vision.py --stage kp --epochs 10 --out models_step4
.venv/bin/python scripts/train_vision.py --stage patch --epochs 15 --out models_step4
.venv/bin/python scripts/train_vision.py --stage kp --epochs 5 --aug 1 --lr 3e-4 --init-from models_step4 --use-val-a --out models
.venv/bin/python scripts/train_vision.py --stage patch --epochs 15 --aug 2 --use-val-a --out models
# (tùy chọn) dựng lại bảng món hàng từ train: scripts/build_nlp_vocab.py
```
Khối luật robot (khối 3) là luật viết tay trong `src/courier/policy/robots.py`, không cần huấn luyện.

## 5. Hệ thống

```
ảnh ──khối 1 (src/courier/vision)──► bản đồ: lưới, đoạn đường + trạng thái (theo chú giải của chính ảnh), bậc thang,
                                      một chiều, địa điểm, robot + hướng mũi, mưa/khô
câu ──khối 2 (src/courier/nlp)─────► nơi giao, điểm ghé, tham chiếu (bản nào), gấp, dễ vỡ
bản đồ + phiếu ──khối 3 (src/courier/policy)──► 10 hướng đi (bảng giá riêng từng robot + tìm đường rẻ nhất)
```
Ghép 3 khối: `src/courier/pipeline.py`. Tổng tham số khi dự đoán ≈ 139,7 triệu (< 200 triệu). Chi tiết: proposal PDF.

## 6. Cấu trúc

```
README.md  requirements.txt
src/courier/   data.py graph.py pipeline.py splits.py | vision/ | nlp/ | policy/
scripts/       predict.py | train_vision.py train_nlp_model.py build_nlp_vocab.py download_models.py
models/        trọng số khối ảnh
submissions/   BEST.json (file dự đoán đã nộp)
```
