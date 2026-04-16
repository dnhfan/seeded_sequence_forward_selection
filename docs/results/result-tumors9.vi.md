# Tumors9 Kết quả và Đánh giá

_Đọc bản tiếng Anh tại [result-tumors9.md](result-tumors9.md)_

[Quay lại mục lục](./README.vi.md)

## 1) EDA (Phân tích khám phá dữ liệu)

- Điểm vào notebook:
- `notebook/Tumors9/01_eda.ipynb`

[Chèn biểu đồ: Tổng quan EDA]
![Tumors9 EDA](../../results/Tumors9/eda/plot/countplot.png)

**Chú thích:**
- Mục đích: Kiểm tra xem bộ dữ liệu có bị mất cân bằng (imbalanced) hay không.
- Cách đọc: Trục hoành (V1) thể hiện các nhãn lớp (0 và 1), trục tung (count) là số lượng mẫu của từng lớp.

## 2) Tiền xử lý dữ liệu

- Điểm vào notebook:
- `notebook/Tumors9/02_preprocess.ipynb`
- Quy ước thư mục đầu ra: `data/processed/Tumors9/01_clean/`

## 3) Lọc đặc trưng (Filter Selection)

- Điểm vào notebook:
- `notebook/Tumors9/03_filter_selection.ipynb`

## 4) Mô hình hóa (so sánh ở giai đoạn filter)

- Điểm vào notebook:
- `notebook/Tumors9/04_modeling.ipynb`
- Kết quả modeling được lưu dưới `results/Tumors9/filter/` khi có sẵn.

## 5) Ensemble Filter (Bỏ phiếu + tập đặc trưng union)

- Điểm vào notebook:
- `notebook/Tumors9/05_ensemble.ipynb`
- Tệp seed pool: `data/processed/Tumors9/03_ensemble/top50_features_voting.csv`
- Kích thước seed pool: 10
- Đặc trưng có số phiếu cao nhất: `V1664(5)`, `V2966(4)`, `V5033(4)`, `V1298(4)`, `V2147(3)`

[Chèn biểu đồ: Bỏ phiếu Ensemble / Đặc trưng Union]
![Tumors9 Ensemble Voting](../../results/Tumors9/ensemble/plots/top50_features_voting.png)

**Chú thích:**
- Mục đích: Hiển thị mức độ đồng thuận của các phương pháp filter khi bỏ phiếu chọn đặc trưng.
- Cách đọc: Trục hoành là tên đặc trưng, trục tung là số phiếu (vote count); đặc trưng có phiếu cao hơn được ưu tiên hơn.

## 6) Wrapper: Sklearn SFS (chạy Raw vs Union)

- Điểm vào script:
- `notebook/Tumors9/06_sklearn_sfs-raw.py`
- `notebook/Tumors9/06_sklearn_sfs-union.py`

| Biến thể | Sklearn Số đặc trưng chọn | Sklearn Global Best | Sklearn Thời gian fit (ms) |
|---|---:|---:|---:|
| Raw | 6 | 0.55 | 538,757 |
| Union | 3 | 0.5 | 14,975 |

## 7) Wrapper: Seeded SFS (chạy Raw vs Union)

- Điểm vào script:
- `notebook/Tumors9/07_sfs-raw.py`
- `notebook/Tumors9/07_sfs-union.py`

| Biến thể | Seeded Số đặc trưng chọn | Seeded Global Best | Seeded Thời gian fit (ms) |
|---|---:|---:|---:|
| Raw | 6 | 0.6 | 38,858 |
| Union | 11 | 0.65 | 4,536 |

## 8) Đánh giá Accuracy (so sánh Raw vs Union)

- Điểm vào notebook:
- `notebook/Tumors9/8_accuracu_evaluate.ipynb`
- `notebook/Tumors9/8_accuracu_evaluate_union.ipynb`

[Chèn biểu đồ: So sánh Accuracy Raw vs Union]
![Tumors9 Accuracy Evaluation](../../results/Tumors9/evaluation/plots/wrapper_sfs_comparison_sk_raw_seeded_raw_Tumors9.png)

**Chú thích:**
- Mục đích: So sánh độ chính xác giữa các cấu hình wrapper (Sklearn SFS và Seeded SFS) theo từng biến thể dữ liệu.
- Cách đọc:
  - Trục hoành là từng cấu hình/phương pháp, trục tung là accuracy; giá trị cao hơn thể hiện hiệu năng tốt hơn.
  - Vạch đen thẳng đứng (Error bar): Thể hiện độ lệch chuẩn (Standard Deviation) qua các fold cross-validation. Vạch này càng ngắn chứng tỏ mô hình dự đoán càng ổn định, ít biến động.
![Tumors9 Accuracy Evaluation](../../results/Tumors9/evaluation/plots/wrapper_sfs_comparison_sk_raw_seeded_union_Tumors9.png)

**Chú thích:**
- Mục đích: So sánh độ chính xác giữa các cấu hình wrapper (Sklearn SFS và Seeded SFS) theo từng biến thể dữ liệu.
- Cách đọc:
  - Trục hoành là từng cấu hình/phương pháp, trục tung là accuracy; giá trị cao hơn thể hiện hiệu năng tốt hơn.
  - Vạch đen thẳng đứng (Error bar): Thể hiện độ lệch chuẩn (Standard Deviation) qua các fold cross-validation. Vạch này càng ngắn chứng tỏ mô hình dự đoán càng ổn định, ít biến động.
![Tumors9 Accuracy Evaluation](../../results/Tumors9/evaluation/plots/wrapper_sfs_comparison_sk_union_seeded_union_Tumors9.png)

**Chú thích:**
- Mục đích: So sánh độ chính xác giữa các cấu hình wrapper (Sklearn SFS và Seeded SFS) theo từng biến thể dữ liệu.
- Cách đọc:
  - Trục hoành là từng cấu hình/phương pháp, trục tung là accuracy; giá trị cao hơn thể hiện hiệu năng tốt hơn.
  - Vạch đen thẳng đứng (Error bar): Thể hiện độ lệch chuẩn (Standard Deviation) qua các fold cross-validation. Vạch này càng ngắn chứng tỏ mô hình dự đoán càng ổn định, ít biến động.

- **Quan sát:** Union seeded cải thiện cả wrapper score và các metric đánh giá cuối so với raw.
- **Giải thích:** Cơ chế gộp union có khả năng làm nổi bật tín hiệu đa lớp vốn yếu.
- **Kết luận:** Đặt union seeded làm mặc định cho Tumors9.

- Cấu hình tốt nhất (raw): `sklearn + Tree`, accuracy trung bình 0.5500, std 0.0236 (2-fold)
- Cấu hình tốt nhất (union): `seeded + LogReg`, accuracy trung bình **0.7333**, std 0.0471 (2-fold)

## 9) Đánh giá thời gian (so sánh thời gian fit Raw vs Union)

- Điểm vào notebook:
- `notebook/Tumors9/9_time_evaluate.ipynb`
- `notebook/Tumors9/9_time_evaluate_union.ipynb`

[Chèn biểu đồ: So sánh thời gian Raw vs Union]
![Tumors9 Time Evaluation](../../results/Tumors9/evaluation/plots/time_comparison_raw_seeded_vs_raw_sklearn.png)

**Chú thích:**
- Mục đích: So sánh chi phí thời gian huấn luyện giữa các phương pháp wrapper trên cùng bộ dữ liệu.
- Cách đọc: Trục hoành là phương pháp/cấu hình, trục tung là tổng thời gian fit (ms); cột thấp hơn nghĩa là chạy nhanh hơn.
![Tumors9 Time Evaluation](../../results/Tumors9/evaluation/plots/time_comparison_union_seeded_vs_union_sklearn.png)

**Chú thích:**
- Mục đích: So sánh chi phí thời gian huấn luyện giữa các phương pháp wrapper trên cùng bộ dữ liệu.
- Cách đọc: Trục hoành là phương pháp/cấu hình, trục tung là tổng thời gian fit (ms); cột thấp hơn nghĩa là chạy nhanh hơn.

- **Quan sát:** Các lần chạy union thường nhanh hơn raw trên hầu hết phương pháp wrapper.
- **Giải thích:** Union làm giảm không gian ứng viên, từ đó giảm tổng số lần fit mô hình.
- **Kết luận:** Dùng union để lặp thử nhanh; dùng raw khi cần tối đa hóa wrapper score.
