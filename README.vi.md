# Pipeline Chọn Đặc Trưng (Filter + Wrapper)

_Read this in [English](README.md)_

Đây là pipeline chọn đặc trưng theo hướng end-to-end, vận hành bằng script, cho nhiều bộ dữ liệu ung thư, gồm:

- Các phương pháp Filter (variance, correlation, chi-squared, mutual information, ANOVA)
- Các phương pháp Wrapper (Sklearn SFS, Seeded SFS)
- Hai biến thể dữ liệu: Raw và Union

**Lưu ý:** _Seeded SFS_ là hướng lai (hybrid), kết hợp giữa Filter + Wrapper.

---

## Hiện Trạng Repository

- Luồng chạy chính là script/notebook trong `notebook/<dataset>/`.
- Quy ước đường dẫn được quản lý bởi `src/config.py` (`ProjectPath`).
- Dữ liệu đã xử lý (machine-readable) nằm trong `data/processed/...`.
- Kết quả thí nghiệm để đọc (human-readable) nằm trong `results/...`.

---

## Các Giai Đoạn Pipeline End-to-End

Mỗi dataset đi qua các bước sau:

1. EDA
2. Preprocess (có thể khác nhau tùy dataset)
3. Filter Selection
4. Modeling (so sánh kết quả ở tầng filter)
5. Ensemble Filter (voting + tạo bộ đặc trưng union)
6. Wrapper: Sklearn SFS (raw/union)
7. Wrapper: Seeded SFS (raw/union)
8. Accuracy Evaluation (raw/union)
9. Time Evaluation (raw/union)

Bố cục dữ liệu xử lý theo chuẩn:

- `data/processed/<dataset>/01_clean`
- `data/processed/<dataset>/02_filter`
- `data/processed/<dataset>/03_ensemble`
- `data/processed/<dataset>/04_wrapper`

---

## Cài Đặt

Chạy từ thư mục gốc repo:

```bash
pip install -r requirements.txt
```

Kiểm tra nhanh dependency:

```bash
python -c "import sklearn, imblearn, seaborn, xgboost"
```

`requirements.txt` được dùng để bao phủ dependency runtime cho luồng end-to-end.

---

## Cách Chạy (End-to-End)

## 1) Chạy các notebook giai đoạn (EDA -> Ensemble)

Với mỗi dataset trong `notebook/<dataset>/`, chạy notebook theo đúng thứ tự giai đoạn.

Ví dụ thư mục dataset:

- `notebook/colon1/`

Thường sẽ có các file:

- `01_eda.ipynb`
- `02_preprocess.ipynb`
- `03_filter_selection.ipynb`
- `04_modeling.ipynb`
- `05_esemble_filter.ipynb`

Tên file chưa đồng nhất hoàn toàn giữa các dataset (ví dụ `8_accuracu...`, `08_accuracy...`, `7_.../8_...` trong `Lung_cancer`).

## 2) Chạy wrapper scripts (raw + union)

Chạy từ repo root (ví dụ: `colon1`):

```bash
python notebook/colon1/06_sklearn_sfs-raw.py
python notebook/colon1/07_sfs-raw.py
python notebook/colon1/06_sklearn_sfs-union.py
python notebook/colon1/07_sfs-union.py
```

Bạn có thể dùng script hỗ trợ:

```bash
bash run_sfs.sh --list
bash run_sfs.sh colon1 raw sklearn
bash run_sfs.sh colon1 raw seeded
bash run_sfs.sh colon1 union sklearn
bash run_sfs.sh colon1 union seeded
```

Chạy batch sklearn wrapper cho tất cả dataset:

```bash
bash run_all_sklearn_sfs.sh
```

## 3) Chạy notebook đánh giá

Mỗi dataset cần chạy:

- Notebook so sánh Accuracy (`8_...evaluate...ipynb` hoặc `08_...`)
- Notebook so sánh Time (`9_time_evaluate...ipynb` hoặc `8_time...` trong `Lung_cancer`)

Cả hai biến thể raw và union đều có sẵn.

---

## Quy Ước Đầu Ra Wrapper

Mỗi lần chạy wrapper được lưu theo cấu trúc:

`results/<dataset>/wrapper/<variant>/<algorithm>/run_YYYYMMDD_HHMMSS[_tag]/`

Mỗi run gồm:

- `history/`
- `features/`
- `metrics/`
- `artifacts/`

Notebook đánh giá thời gian sử dụng:

- `metrics/metrics.csv` với cột `total_fit_time_ms`

---

## Quy Tắc Biến Thể (Raw vs Union)

- Tách biệt nghiêm ngặt artifact raw và union bằng `dataset_variant` (`raw`, `union`).
- Tên output cần có tag biến thể để tránh ghi đè.
- Mẫu đặt tên trong notebook so sánh Accuracy:

```python
experiment_prefix=f"wrapper_sfs_comparison_sk_{sk_data_variant}_seeded_{data_variant}"
chart_title=f"Sklearn sfs({sk_data_variant}) vs Seeded sfs({data_variant}) Performance"
```

---

## Các Điểm Vào Quan Trọng Trong Source

- Quy ước đường dẫn và thư mục: `src/config.py` (`ProjectPath`)
- Cấu trúc output cho wrapper run: `src/utils/experiment_paths.py`, `src/wrapper/base.py`
- Logic đánh giá: `src/modeling/evaluation.py`
- Tạo union feature: `src/utils/utils.py` (`create_union_features`)

---

## Lệnh Kiểm Tra Nhanh

Kiểm tra syntax:

```bash
python -m py_compile src/modeling/evaluation.py
```

Kiểm tra metrics của union wrapper tồn tại:

```bash
python - <<'PY'
from pathlib import Path
p = Path("results/colon1/wrapper/union")
print(p.exists(), list(p.glob("*/*/metrics/metrics.csv"))[:3])
PY
```

---

## Lưu Ý Quan Trọng

- Tên notebook không đồng nhất giữa các dataset; giữ đúng convention của từng thư mục.
- Nhiều notebook được sửa trực tiếp dạng JSON; tránh thay đổi metadata/id không cần thiết.
- Wrapper scripts phụ thuộc `sys.path` injection; nên chạy từ repo root.

---

## Mục Lục Tài Liệu
