# CNS1 Kết quả và Đánh giá

_Đọc bản tiếng Anh tại [result-cns1.md](result-cns1.md)_

[Quay lại mục lục](./README.vi.md)

## 1) EDA (Phân tích khám phá dữ liệu)

- Điểm vào notebook:
- `notebook/CNS1/01_eda.ipynb`

[Chèn biểu đồ: Tổng quan EDA]
![CNS1 EDA](../../results/CNS1/eda/plot/countplot.png)

**Chú thích:**
- Mục đích: Kiểm tra xem bộ dữ liệu có bị mất cân bằng (imbalanced) hay không.
- Cách đọc: Trục hoành (V1) thể hiện các nhãn lớp (0 và 1), trục tung (count) là số lượng mẫu của từng lớp.

## 2) Tiền xử lý dữ liệu

- Điểm vào notebook:
- `notebook/CNS1/02_preprocess.ipynb`
- Quy ước thư mục đầu ra: `data/processed/CNS1/01_clean/`

## 3) Lọc đặc trưng (Filter Selection)

- Điểm vào notebook:
- `notebook/CNS1/03_filter_selection.ipynb`
- Tệp báo cáo: `results/CNS1/filter/reports/evaluation_CNS1.txt`

[Chèn biểu đồ: So sánh Filter Selection]
![CNS1 Filter Selection](../../results/CNS1/filter/plots/evaluation_CNS1.png)

**Chú thích:**
- Mục đích: So sánh hiệu năng các phương pháp filter để chọn ra nhóm đặc trưng tốt nhất cho bước tiếp theo.
- Cách đọc: Trục hoành là các phương pháp filter, trục tung là điểm đánh giá; cột/điểm càng cao thì phương pháp càng tốt.

## 4) Mô hình hóa (so sánh ở giai đoạn filter)

- Điểm vào notebook:
- `notebook/CNS1/04_modeling.ipynb`
- Kết quả modeling được lưu dưới `results/CNS1/filter/` khi có sẵn.

## 5) Ensemble Filter (Bỏ phiếu + tập đặc trưng union)

- Điểm vào notebook:
- `notebook/CNS1/05_esemble_filter.ipynb`
- Tệp seed pool: `data/processed/CNS1/03_ensemble/top50_features_voting.csv`
- Kích thước seed pool: 10
- Đặc trưng có số phiếu cao nhất: `1053(4)`, `6041(4)`, `4307(4)`, `1477(4)`, `653(4)`

[Chèn biểu đồ: Bỏ phiếu Ensemble / Đặc trưng Union]
![CNS1 Ensemble Voting](../../results/CNS1/ensemble/plots/top50_features_voting.png)

**Chú thích:**
- Mục đích: Hiển thị mức độ đồng thuận của các phương pháp filter khi bỏ phiếu chọn đặc trưng.
- Cách đọc: Trục hoành là tên đặc trưng, trục tung là số phiếu (vote count); đặc trưng có phiếu cao hơn được ưu tiên hơn.

## 6) Wrapper: Sklearn SFS (chạy Raw vs Union)

- Điểm vào script:
- `notebook/CNS1/06_sklearn_sfs-raw.py`
- `notebook/CNS1/06_sklearn_sfs-union.py`

| Biến thể | Sklearn Số đặc trưng chọn | Sklearn Global Best | Sklearn Thời gian fit (ms) |
|---|---:|---:|---:|
| Raw | 6 | 0.9167 | 801,554 |
| Union | 4 | 0.9 | 63,018 |

## 7) Wrapper: Seeded SFS (chạy Raw vs Union)

- Điểm vào script:
- `notebook/CNS1/07_sfs-raw.py`
- `notebook/CNS1/07_sfs-union.py`

| Biến thể | Seeded Số đặc trưng chọn | Seeded Global Best | Seeded Thời gian fit (ms) |
|---|---:|---:|---:|
| Raw | 9 | 0.95 | 484,297 |
| Union | 5 | 0.9167 | 22,330 |

## 8) Đánh giá Accuracy (so sánh Raw vs Union)

- Điểm vào notebook:
- `notebook/CNS1/8_accuracu_evaluate.ipynb`
- `notebook/CNS1/8_accuracu_evaluate_union.ipynb`

[Chèn biểu đồ: So sánh Accuracy Raw vs Union]
![CNS1 Accuracy Evaluation](../../results/CNS1/evaluation/plots/wrapper_sfs_comparison_sk_raw_seeded_raw_CNS1.png)

**Chú thích:**
- Mục đích: So sánh độ chính xác giữa các cấu hình wrapper (Sklearn SFS và Seeded SFS) theo từng biến thể dữ liệu.
- Cách đọc:
  - Trục hoành là từng cấu hình/phương pháp, trục tung là accuracy; giá trị cao hơn thể hiện hiệu năng tốt hơn.
  - Vạch đen thẳng đứng (Error bar): Thể hiện độ lệch chuẩn (Standard Deviation) qua các fold cross-validation. Vạch này càng ngắn chứng tỏ mô hình dự đoán càng ổn định, ít biến động.
![CNS1 Accuracy Evaluation](../../results/CNS1/evaluation/plots/wrapper_sfs_comparison_sk_union_seeded_union_CNS1.png)

**Chú thích:**
- Mục đích: So sánh độ chính xác giữa các cấu hình wrapper (Sklearn SFS và Seeded SFS) theo từng biến thể dữ liệu.
- Cách đọc:
  - Trục hoành là từng cấu hình/phương pháp, trục tung là accuracy; giá trị cao hơn thể hiện hiệu năng tốt hơn.
  - Vạch đen thẳng đứng (Error bar): Thể hiện độ lệch chuẩn (Standard Deviation) qua các fold cross-validation. Vạch này càng ngắn chứng tỏ mô hình dự đoán càng ổn định, ít biến động.

- **Quan sát:** Raw seeded đạt wrapper score cao hơn nhưng cần thời gian chạy dài hơn nhiều.
- **Giải thích:** Tìm kiếm trên raw đánh giá không gian ứng viên lớn hơn union.
- **Kết luận:** Chọn biến thể theo mục tiêu: raw để tối đa hóa điểm số, union để tối ưu hiệu quả.

- Cấu hình tốt nhất (raw): `sklearn + LogReg`, accuracy trung bình 0.8833, std 0.0456
- Cấu hình tốt nhất (union): `sklearn + LogReg`, accuracy trung bình 0.8833, std 0.0950

## 9) Đánh giá thời gian (so sánh thời gian fit Raw vs Union)

- Điểm vào notebook:
- `notebook/CNS1/9_time_evaluate.ipynb`
- `notebook/CNS1/9_time_evaluate_union.ipynb`

[Chèn biểu đồ: So sánh thời gian Raw vs Union]
![CNS1 Time Evaluation](../../results/CNS1/evaluation/plots/time_comparison_raw_seeded_vs_raw_sklearn.png)

**Chú thích:**
- Mục đích: So sánh chi phí thời gian huấn luyện giữa các phương pháp wrapper trên cùng bộ dữ liệu.
- Cách đọc: Trục hoành là phương pháp/cấu hình, trục tung là tổng thời gian fit (ms); cột thấp hơn nghĩa là chạy nhanh hơn.
![CNS1 Time Evaluation](../../results/CNS1/evaluation/plots/time_comparison_union_seeded_vs_union_sklearn.png)

**Chú thích:**
- Mục đích: So sánh chi phí thời gian huấn luyện giữa các phương pháp wrapper trên cùng bộ dữ liệu.
- Cách đọc: Trục hoành là phương pháp/cấu hình, trục tung là tổng thời gian fit (ms); cột thấp hơn nghĩa là chạy nhanh hơn.

- **Quan sát:** Các lần chạy union thường nhanh hơn raw trên hầu hết phương pháp wrapper.
- **Giải thích:** Union làm giảm không gian ứng viên, từ đó giảm tổng số lần fit mô hình.
- **Kết luận:** Dùng union để lặp thử nhanh; dùng raw khi cần tối đa hóa wrapper score.
