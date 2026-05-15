# SRBCT_txt Kết quả và Đánh giá

_Đọc bản tiếng Anh tại [result-srbct_txt.md](result-srbct_txt.md)_

[Quay lại mục lục](./README.vi.md)

## 1) EDA (Phân tích khám phá dữ liệu)

- Điểm vào notebook:
- `notebook/SRBCT_txt/01_eda.ipynb`

[Chèn biểu đồ: Tổng quan EDA]
![SRBCT_txt EDA](../../results/SRBCT_txt/eda/plot/countplot.png)

**Chú thích:**
- Mục đích: Kiểm tra xem bộ dữ liệu có bị mất cân bằng (imbalanced) hay không.
- Cách đọc: Trục hoành (V1) thể hiện các nhãn lớp (0 và 1), trục tung (count) là số lượng mẫu của từng lớp.

## 2) Tiền xử lý dữ liệu

- Điểm vào notebook:
- `notebook/SRBCT_txt/02_preprocess.ipynb`
- Quy ước thư mục đầu ra: `data/processed/SRBCT_txt/01_clean/`

## 3) Lọc đặc trưng (Filter Selection)

- Điểm vào notebook:
- `notebook/SRBCT_txt/03_filter_selection.ipynb`
- Tệp báo cáo: `results/SRBCT_txt/filter/reports/evaluation_SRBCT_txt.txt`

[Chèn biểu đồ: So sánh Filter Selection]
![SRBCT_txt Filter Selection](../../results/SRBCT_txt/filter/plots/evaluation_SRBCT_txt.png)

**Chú thích:**
- Mục đích: So sánh hiệu năng các phương pháp filter để chọn ra nhóm đặc trưng tốt nhất cho bước tiếp theo.
- Cách đọc: Trục hoành là các phương pháp filter, trục tung là điểm đánh giá; cột/điểm càng cao thì phương pháp càng tốt.

## 4) Mô hình hóa (so sánh ở giai đoạn filter)

- Điểm vào notebook:
- `notebook/SRBCT_txt/04_modeling.ipynb`
- Kết quả modeling được lưu dưới `results/SRBCT_txt/filter/` khi có sẵn.

## 5) Ensemble Filter (Bỏ phiếu + tập đặc trưng union)

- Điểm vào notebook:
- `notebook/SRBCT_txt/05_ensemble.ipynb`
- Tệp seed pool: `data/processed/SRBCT_txt/03_ensemble/top50_features_voting.csv`
- Kích thước seed pool: 10
- Đặc trưng có số phiếu cao nhất: `V510(5)`, `V546(5)`, `V247(5)`, `V1956(5)`, `V1955(5)`

[Chèn biểu đồ: Bỏ phiếu Ensemble / Đặc trưng Union]
![SRBCT_txt Ensemble Voting](../../results/SRBCT_txt/ensemble/plots/top50_features_voting.png)

**Chú thích:**
- Mục đích: Hiển thị mức độ đồng thuận của các phương pháp filter khi bỏ phiếu chọn đặc trưng.
- Cách đọc: Trục hoành là tên đặc trưng, trục tung là số phiếu (vote count); đặc trưng có phiếu cao hơn được ưu tiên hơn.

## 6) Wrapper: Sklearn SFS (chạy Raw vs Union)

- Điểm vào script:
- `notebook/SRBCT_txt/06_sklearn_sfs-raw.py`
- `notebook/SRBCT_txt/06_sklearn_sfs-union.py`

| Biến thể | Sklearn Số đặc trưng chọn | Sklearn Global Best | Sklearn Thời gian fit (ms) |
|---|---:|---:|---:|
| Raw | 4 | 1 | 151,507 |
| Union | 4 | 1 | 11,372 |

## 7) Wrapper: Seeded SFS (chạy Raw vs Union)

- Điểm vào script:
- `notebook/SRBCT_txt/07_sfs-raw.py`
- `notebook/SRBCT_txt/07_sfs-union.py`

| Biến thể | Seeded Số đặc trưng chọn | Seeded Global Best | Seeded Thời gian fit (ms) |
|---|---:|---:|---:|
| Raw | 4 | 1 | 38,541 |
| Union | 4 | 1 | 4,077 |

## 8) Đánh giá Accuracy (so sánh Raw vs Union)

- Điểm vào notebook:
- `notebook/SRBCT_txt/8_accuracu_evaluate.ipynb`
- `notebook/SRBCT_txt/8_accuracu_evaluate_union.ipynb`

[Chèn biểu đồ: So sánh Accuracy Raw vs Union]
![SRBCT_txt Accuracy Evaluation](../../results/SRBCT_txt/evaluation/plots/wrapper_sfs_comparison_sk_raw_seeded_raw_SRBCT_txt.png)

**Chú thích:**
- Mục đích: So sánh độ chính xác giữa các cấu hình wrapper (Sklearn SFS và Seeded SFS) theo từng biến thể dữ liệu.
- Cách đọc:
  - Trục hoành là từng cấu hình/phương pháp, trục tung là accuracy; giá trị cao hơn thể hiện hiệu năng tốt hơn.
  - Vạch đen thẳng đứng (Error bar): Thể hiện độ lệch chuẩn (Standard Deviation) qua các fold cross-validation. Vạch này càng ngắn chứng tỏ mô hình dự đoán càng ổn định, ít biến động.
![SRBCT_txt Accuracy Evaluation](../../results/SRBCT_txt/evaluation/plots/wrapper_sfs_comparison_sk_union_seeded_union_SRBCT_txt.png)

**Chú thích:**
- Mục đích: So sánh độ chính xác giữa các cấu hình wrapper (Sklearn SFS và Seeded SFS) theo từng biến thể dữ liệu.
- Cách đọc:
  - Trục hoành là từng cấu hình/phương pháp, trục tung là accuracy; giá trị cao hơn thể hiện hiệu năng tốt hơn.
  - Vạch đen thẳng đứng (Error bar): Thể hiện độ lệch chuẩn (Standard Deviation) qua các fold cross-validation. Vạch này càng ngắn chứng tỏ mô hình dự đoán càng ổn định, ít biến động.

- **Quan sát:** Accuracy giữ ổn định trong khi thời gian chạy của union giảm đáng kể.
- **Giải thích:** Lọc union không loại bỏ các đặc trưng dự đoán cốt lõi của bộ dữ liệu này.
- **Kết luận:** Dùng union seeded để rút ngắn chu kỳ mà không giảm chất lượng.

- Cấu hình tốt nhất (raw): `seeded + LogReg`, accuracy trung bình **0.9846**, std 0.0344
- Cấu hình tốt nhất (union): `seeded + LogReg`, accuracy trung bình **0.9846**, std 0.0344

## 9) Đánh giá thời gian (so sánh thời gian fit Raw vs Union)

- Điểm vào notebook:
- `notebook/SRBCT_txt/9_time_evaluate.ipynb`
- `notebook/SRBCT_txt/9_time_evaluate_union.ipynb`

[Chèn biểu đồ: So sánh thời gian Raw vs Union]
![SRBCT_txt Time Evaluation](../../results/SRBCT_txt/evaluation/plots/time_comparison_raw_seeded_vs_raw_sklearn.png)

**Chú thích:**
- Mục đích: So sánh chi phí thời gian huấn luyện giữa các phương pháp wrapper trên cùng bộ dữ liệu.
- Cách đọc: Trục hoành là phương pháp/cấu hình, trục tung là tổng thời gian fit (ms); cột thấp hơn nghĩa là chạy nhanh hơn.
![SRBCT_txt Time Evaluation](../../results/SRBCT_txt/evaluation/plots/time_comparison_union_seeded_vs_union_sklearn.png)

**Chú thích:**
- Mục đích: So sánh chi phí thời gian huấn luyện giữa các phương pháp wrapper trên cùng bộ dữ liệu.
- Cách đọc: Trục hoành là phương pháp/cấu hình, trục tung là tổng thời gian fit (ms); cột thấp hơn nghĩa là chạy nhanh hơn.

- **Quan sát:** Các lần chạy union thường nhanh hơn raw trên hầu hết phương pháp wrapper.
- **Giải thích:** Union làm giảm không gian ứng viên, từ đó giảm tổng số lần fit mô hình.
- **Kết luận:** Dùng union để lặp thử nhanh; dùng raw khi cần tối đa hóa wrapper score.


## 10) Đánh Giá Cuối Cùng (So Sánh Tất Cả Phương Pháp)

- Điểm vào notebook:
- `notebook/SRBCT_txt/10_final_evaluate.ipynb`
- Báo cáo: `results/SRBCT_txt/evaluation/reports/final_evaluation_all_methods_srbct_txt_SRBCT_txt.txt`

[Biểu Đồ: Đánh Giá Cuối Cùng - Tất Cả Phương Pháp]
![SRBCT_txt Final Evaluation](../../results/SRBCT_txt/evaluation/plots/final_evaluation_all_methods_srbct_txt_SRBCT_txt.png)

**Chú Thích:**
- Mục đích: So sánh tất cả phương pháp lựa chọn đặc trưng (Filter, Ensemble, Sklearn SFS, Seeded SFS) với cả hai mô hình LogReg và Tree.
- Cách đọc:
  - Trục X liệt kê tất cả các kết hợp phương pháp/mô hình (ví dụ: "Sklearn_SFS_Raw + LogReg").
  - Trục Y hiển thị độ chính xác cross-validation; các cột cao hơn cho biết hiệu suất tốt hơn.
  - Các thanh lỗi dọc hiển thị độ lệch chuẩn (Std) trên các fold; các thanh ngắn hơn chỉ ra mô hình ổn định hơn.

| Xếp Hạng | Phương Pháp + Mô Hình | CV Fold | Accuracy Trung Bình | Std | Median | Min | Max |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 | ANOVA_F_TEST + LogReg | 5 | **1.0000** | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| 2 | CHI_SQUARED + LogReg | 5 | **1.0000** | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| 3 | MUTUAL_INFORMATION + LogReg | 5 | **1.0000** | 0.0000 | 1.0000 | 1.0000 | 1.0000 |

**Quan Sát Chính:**
- Cấu hình tốt nhất: ANOVA_F_TEST + LogReg với độ chính xác 1.0000 (σ=0.0000)
- Xếp thứ hai: CHI_SQUARED + LogReg với độ chính xác 1.0000
- Khuyến nghị: Xem so sánh chi tiết trong biểu đồ và tệp báo cáo ở trên.