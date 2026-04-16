# DLBCL Kết quả và Đánh giá

_Đọc bản tiếng Anh tại [result-dlbcl.md](result-dlbcl.md)_

[Quay lại mục lục](./README.vi.md)

## 1) EDA (Phân tích khám phá dữ liệu)

- Điểm vào notebook:
- `notebook/DLBCL/01_eda.ipynb`

[Chèn biểu đồ: Tổng quan EDA]
![DLBCL EDA](../../results/DLBCL/eda/plot/countplot.png)

**Chú thích:**
- Mục đích: Kiểm tra xem bộ dữ liệu có bị mất cân bằng (imbalanced) hay không.
- Cách đọc: Trục hoành (V1) thể hiện các nhãn lớp (0 và 1), trục tung (count) là số lượng mẫu của từng lớp.

## 2) Tiền xử lý dữ liệu

- Điểm vào notebook:
- `notebook/DLBCL/02_preprocess.ipynb`
- Quy ước thư mục đầu ra: `data/processed/DLBCL/01_clean/`

## 3) Lọc đặc trưng (Filter Selection)

- Điểm vào notebook:
- `notebook/DLBCL/03_filter_selection.ipynb`
- Tệp báo cáo: `results/DLBCL/filter/reports/evaluation_DLBCL.txt`

[Chèn biểu đồ: So sánh Filter Selection]
![DLBCL Filter Selection](../../results/DLBCL/filter/plots/evaluation_DLBCL.png)

**Chú thích:**
- Mục đích: So sánh hiệu năng các phương pháp filter để chọn ra nhóm đặc trưng tốt nhất cho bước tiếp theo.
- Cách đọc: Trục hoành là các phương pháp filter, trục tung là điểm đánh giá; cột/điểm càng cao thì phương pháp càng tốt.

## 4) Mô hình hóa (so sánh ở giai đoạn filter)

- Điểm vào notebook:
- `notebook/DLBCL/04_modeling.ipynb`
- Kết quả modeling được lưu dưới `results/DLBCL/filter/` khi có sẵn.

## 5) Ensemble Filter (Bỏ phiếu + tập đặc trưng union)

- Điểm vào notebook:
- `notebook/DLBCL/05_esemble_filter.ipynb`
- Tệp seed pool: `data/processed/DLBCL/03_ensemble/top50_features_voting.csv`
- Kích thước seed pool: 10
- Đặc trưng có số phiếu cao nhất: `V3128(5)`, `V4553(5)`, `V1056(4)`, `V1601(4)`, `V3468(4)`

[Chèn biểu đồ: Bỏ phiếu Ensemble / Đặc trưng Union]
![DLBCL Ensemble Voting](../../results/DLBCL/ensemble/plots/top50_features_voting.png)

**Chú thích:**
- Mục đích: Hiển thị mức độ đồng thuận của các phương pháp filter khi bỏ phiếu chọn đặc trưng.
- Cách đọc: Trục hoành là tên đặc trưng, trục tung là số phiếu (vote count); đặc trưng có phiếu cao hơn được ưu tiên hơn.

## 6) Wrapper: Sklearn SFS (chạy Raw vs Union)

- Điểm vào script:
- `notebook/DLBCL/06_sklearn_sfs-raw.py`
- `notebook/DLBCL/06_sklearn_sfs-union.py`

| Biến thể | Sklearn Số đặc trưng chọn | Sklearn Global Best | Sklearn Thời gian fit (ms) |
|---|---:|---:|---:|
| Raw | 2 | 1 | 229,964 |
| Union | 3 | 1 | 13,010 |

## 7) Wrapper: Seeded SFS (chạy Raw vs Union)

- Điểm vào script:
- `notebook/DLBCL/07_sfs-raw.py`
- `notebook/DLBCL/07_sfs-union.py`

| Biến thể | Seeded Số đặc trưng chọn | Seeded Global Best | Seeded Thời gian fit (ms) |
|---|---:|---:|---:|
| Raw | 3 | 0.9867 | 104,516 |
| Union | 3 | 0.9742 | 6,687 |

## 8) Đánh giá Accuracy (so sánh Raw vs Union)

- Điểm vào notebook:
- `notebook/DLBCL/8_accuracu_evaluate.ipynb`
- `notebook/DLBCL/8_accuracu_evaluate_union.ipynb`

[Chèn biểu đồ: So sánh Accuracy Raw vs Union]
![DLBCL Accuracy Evaluation](../../results/DLBCL/evaluation/plots/wrapper_sfs_comparison_DLBCL.png)

**Chú thích:**
- Mục đích: So sánh độ chính xác giữa các cấu hình wrapper (Sklearn SFS và Seeded SFS) theo từng biến thể dữ liệu.
- Cách đọc:
  - Trục hoành là từng cấu hình/phương pháp, trục tung là accuracy; giá trị cao hơn thể hiện hiệu năng tốt hơn.
  - Vạch đen thẳng đứng (Error bar): Thể hiện độ lệch chuẩn (Standard Deviation) qua các fold cross-validation. Vạch này càng ngắn chứng tỏ mô hình dự đoán càng ổn định, ít biến động.
![DLBCL Accuracy Evaluation](../../results/DLBCL/evaluation/plots/wrapper_sfs_comparison_sk_raw_seeded_raw_DLBCL.png)

**Chú thích:**
- Mục đích: So sánh độ chính xác giữa các cấu hình wrapper (Sklearn SFS và Seeded SFS) theo từng biến thể dữ liệu.
- Cách đọc:
  - Trục hoành là từng cấu hình/phương pháp, trục tung là accuracy; giá trị cao hơn thể hiện hiệu năng tốt hơn.
  - Vạch đen thẳng đứng (Error bar): Thể hiện độ lệch chuẩn (Standard Deviation) qua các fold cross-validation. Vạch này càng ngắn chứng tỏ mô hình dự đoán càng ổn định, ít biến động.
![DLBCL Accuracy Evaluation](../../results/DLBCL/evaluation/plots/wrapper_sfs_comparison_sk_union_seeded_union_DLBCL.png)

**Chú thích:**
- Mục đích: So sánh độ chính xác giữa các cấu hình wrapper (Sklearn SFS và Seeded SFS) theo từng biến thể dữ liệu.
- Cách đọc:
  - Trục hoành là từng cấu hình/phương pháp, trục tung là accuracy; giá trị cao hơn thể hiện hiệu năng tốt hơn.
  - Vạch đen thẳng đứng (Error bar): Thể hiện độ lệch chuẩn (Standard Deviation) qua các fold cross-validation. Vạch này càng ngắn chứng tỏ mô hình dự đoán càng ổn định, ít biến động.

- **Quan sát:** Hiệu năng gần mức trần đạt được chỉ với 2-3 đặc trưng.
- **Giải thích:** Tín hiệu dự đoán tập trung trong một tập con đặc trưng gọn.
- **Kết luận:** Ưu tiên tập đặc trưng tối giản để tăng khả năng diễn giải và đơn giản vận hành.

- Cấu hình tốt nhất (raw): `seeded + LogReg`, accuracy trung bình **0.9867**, std 0.0298
- Cấu hình tốt nhất (union): `seeded + LogReg`, accuracy trung bình 0.9492, std 0.0817
- Final selected features (winning setup, raw seeded): `V3128, V454, V120`

## 9) Đánh giá thời gian (so sánh thời gian fit Raw vs Union)

- Điểm vào notebook:
- `notebook/DLBCL/9_time_evaluate.ipynb`
- `notebook/DLBCL/9_time_evaluate_union.ipynb`

[Chèn biểu đồ: So sánh thời gian Raw vs Union]
![DLBCL Time Evaluation](../../results/DLBCL/evaluation/plots/time_comparison_raw_seeded_vs_raw_sklearn.png)

**Chú thích:**
- Mục đích: So sánh chi phí thời gian huấn luyện giữa các phương pháp wrapper trên cùng bộ dữ liệu.
- Cách đọc: Trục hoành là phương pháp/cấu hình, trục tung là tổng thời gian fit (ms); cột thấp hơn nghĩa là chạy nhanh hơn.
![DLBCL Time Evaluation](../../results/DLBCL/evaluation/plots/time_comparison_seeded3_vs_sklearn_brain.png)

**Chú thích:**
- Mục đích: So sánh chi phí thời gian huấn luyện giữa các phương pháp wrapper trên cùng bộ dữ liệu.
- Cách đọc: Trục hoành là phương pháp/cấu hình, trục tung là tổng thời gian fit (ms); cột thấp hơn nghĩa là chạy nhanh hơn.
![DLBCL Time Evaluation](../../results/DLBCL/evaluation/plots/time_comparison_union_seeded_vs_union_sklearn.png)

**Chú thích:**
- Mục đích: So sánh chi phí thời gian huấn luyện giữa các phương pháp wrapper trên cùng bộ dữ liệu.
- Cách đọc: Trục hoành là phương pháp/cấu hình, trục tung là tổng thời gian fit (ms); cột thấp hơn nghĩa là chạy nhanh hơn.

- **Quan sát:** Các lần chạy union thường nhanh hơn raw trên hầu hết phương pháp wrapper.
- **Giải thích:** Union làm giảm không gian ứng viên, từ đó giảm tổng số lần fit mô hình.
- **Kết luận:** Dùng union để lặp thử nhanh; dùng raw khi cần tối đa hóa wrapper score.
