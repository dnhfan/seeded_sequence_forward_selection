# Lymphoma Kết quả và Đánh giá

_Đọc bản tiếng Anh tại [result-lymphoma.md](result-lymphoma.md)_

[Quay lại mục lục](./README.vi.md)

## 1) EDA (Phân tích khám phá dữ liệu)

- Điểm vào notebook:
- `notebook/Lymphoma/01_eda.ipynb`

[Chèn biểu đồ: Tổng quan EDA]
![Lymphoma EDA](../../results/Lymphoma/eda/plot/countplot.png)

**Chú thích:**
- Mục đích: Kiểm tra xem bộ dữ liệu có bị mất cân bằng (imbalanced) hay không.
- Cách đọc: Trục hoành (V1) thể hiện các nhãn lớp (0 và 1), trục tung (count) là số lượng mẫu của từng lớp.

## 2) Tiền xử lý dữ liệu

- Điểm vào notebook:
- `notebook/Lymphoma/02_Preprocessing.ipynb`
- Quy ước thư mục đầu ra: `data/processed/Lymphoma/01_clean/`

## 3) Lọc đặc trưng (Filter Selection)

- Điểm vào notebook:
- `notebook/Lymphoma/03_filter_selection.ipynb`
- Tệp báo cáo: `results/Lymphoma/filter/reports/evaluation_Lymphoma.txt`

[Chèn biểu đồ: So sánh Filter Selection]
![Lymphoma Filter Selection](../../results/Lymphoma/filter/plots/evaluation_Lymphoma.png)

**Chú thích:**
- Mục đích: So sánh hiệu năng các phương pháp filter để chọn ra nhóm đặc trưng tốt nhất cho bước tiếp theo.
- Cách đọc: Trục hoành là các phương pháp filter, trục tung là điểm đánh giá; cột/điểm càng cao thì phương pháp càng tốt.

## 4) Mô hình hóa (so sánh ở giai đoạn filter)

- Điểm vào notebook:
- `notebook/Lymphoma/04_modeling.ipynb`
- Kết quả modeling được lưu dưới `results/Lymphoma/filter/` khi có sẵn.

## 5) Ensemble Filter (Bỏ phiếu + tập đặc trưng union)

- Điểm vào notebook:
- `notebook/Lymphoma/05_esemble_filter.ipynb`
- Tệp seed pool: `data/processed/Lymphoma/03_ensemble/top50_features_voting.csv`
- Kích thước seed pool: 10
- Đặc trưng có số phiếu cao nhất: `V3755(5)`, `V3790(5)`, `V3783(4)`, `V3764(4)`, `V758(4)`

[Chèn biểu đồ: Bỏ phiếu Ensemble / Đặc trưng Union]
![Lymphoma Ensemble Voting](../../results/Lymphoma/ensemble/plots/top50_features_voting.png)

**Chú thích:**
- Mục đích: Hiển thị mức độ đồng thuận của các phương pháp filter khi bỏ phiếu chọn đặc trưng.
- Cách đọc: Trục hoành là tên đặc trưng, trục tung là số phiếu (vote count); đặc trưng có phiếu cao hơn được ưu tiên hơn.

## 6) Wrapper: Sklearn SFS (chạy Raw vs Union)

- Điểm vào script:
- `notebook/Lymphoma/06_sklearn_sfs-raw.py`
- `notebook/Lymphoma/06_sklearn_sfs-union.py`

| Biến thể | Sklearn Số đặc trưng chọn | Sklearn Global Best | Sklearn Thời gian fit (ms) |
|---|---:|---:|---:|
| Raw | 3 | 1 | 209,993 |
| Union | 3 | 1 | 13,759 |

## 7) Wrapper: Seeded SFS (chạy Raw vs Union)

- Điểm vào script:
- `notebook/Lymphoma/07_sfs-raw.py`
- `notebook/Lymphoma/07_sfs-union.py`

| Biến thể | Seeded Số đặc trưng chọn | Seeded Global Best | Seeded Thời gian fit (ms) |
|---|---:|---:|---:|
| Raw | 2 | 1 | 35,799 |
| Union | 2 | 1 | 3,496 |

## 8) Đánh giá Accuracy (so sánh Raw vs Union)

- Điểm vào notebook:
- `notebook/Lymphoma/8_accuracu_evaluate.ipynb`
- `notebook/Lymphoma/8_accuracu_evaluate_union.ipynb`

[Chèn biểu đồ: So sánh Accuracy Raw vs Union]
![Lymphoma Accuracy Evaluation](../../results/Lymphoma/evaluation/plots/wrapper_sfs_comparison_sk_raw_seeded_raw_Lymphoma.png)

**Chú thích:**
- Mục đích: So sánh độ chính xác giữa các cấu hình wrapper (Sklearn SFS và Seeded SFS) theo từng biến thể dữ liệu.
- Cách đọc:
  - Trục hoành là từng cấu hình/phương pháp, trục tung là accuracy; giá trị cao hơn thể hiện hiệu năng tốt hơn.
  - Vạch đen thẳng đứng (Error bar): Thể hiện độ lệch chuẩn (Standard Deviation) qua các fold cross-validation. Vạch này càng ngắn chứng tỏ mô hình dự đoán càng ổn định, ít biến động.
![Lymphoma Accuracy Evaluation](../../results/Lymphoma/evaluation/plots/wrapper_sfs_comparison_sk_union_seeded_union_Lymphoma.png)

**Chú thích:**
- Mục đích: So sánh độ chính xác giữa các cấu hình wrapper (Sklearn SFS và Seeded SFS) theo từng biến thể dữ liệu.
- Cách đọc:
  - Trục hoành là từng cấu hình/phương pháp, trục tung là accuracy; giá trị cao hơn thể hiện hiệu năng tốt hơn.
  - Vạch đen thẳng đứng (Error bar): Thể hiện độ lệch chuẩn (Standard Deviation) qua các fold cross-validation. Vạch này càng ngắn chứng tỏ mô hình dự đoán càng ổn định, ít biến động.

- **Quan sát:** Các phương pháp dẫn đầu đạt hiệu năng gần mức trần ở cả hai biến thể.
- **Giải thích:** Khả năng phân tách lớp vẫn mạnh sau cả pipeline raw và union.
- **Kết luận:** Ưu tiên union seeded khi cần accuracy tương đương với chi phí thấp hơn.

- Cấu hình tốt nhất (raw): `sklearn + LogReg`, accuracy trung bình **1.0000**, std 0.0000
- Cấu hình tốt nhất (union): `seeded + LogReg`, accuracy trung bình **1.0000**, std 0.0000

## 9) Đánh giá thời gian (so sánh thời gian fit Raw vs Union)

- Điểm vào notebook:
- `notebook/Lymphoma/9_time_evaluate.ipynb`
- `notebook/Lymphoma/9_time_evaluate_union.ipynb`

[Chèn biểu đồ: So sánh thời gian Raw vs Union]
![Lymphoma Time Evaluation](../../results/Lymphoma/evaluation/plots/time_comparison_seeded_vs_sklearn.png)

**Chú thích:**
- Mục đích: So sánh chi phí thời gian huấn luyện giữa các phương pháp wrapper trên cùng bộ dữ liệu.
- Cách đọc: Trục hoành là phương pháp/cấu hình, trục tung là tổng thời gian fit (ms); cột thấp hơn nghĩa là chạy nhanh hơn.
![Lymphoma Time Evaluation](../../results/Lymphoma/evaluation/plots/time_comparison_union_seeded_vs_union_sklearn.png)

**Chú thích:**
- Mục đích: So sánh chi phí thời gian huấn luyện giữa các phương pháp wrapper trên cùng bộ dữ liệu.
- Cách đọc: Trục hoành là phương pháp/cấu hình, trục tung là tổng thời gian fit (ms); cột thấp hơn nghĩa là chạy nhanh hơn.

- **Quan sát:** Các lần chạy union thường nhanh hơn raw trên hầu hết phương pháp wrapper.
- **Giải thích:** Union làm giảm không gian ứng viên, từ đó giảm tổng số lần fit mô hình.
- **Kết luận:** Dùng union để lặp thử nhanh; dùng raw khi cần tối đa hóa wrapper score.
