# Prostate Kết quả và Đánh giá

_Đọc bản tiếng Anh tại [result-prostate.md](result-prostate.md)_

[Quay lại mục lục](./README.vi.md)

## 1) EDA (Phân tích khám phá dữ liệu)

- Điểm vào notebook:
- `notebook/Prostate/01_eda.ipynb`

[Chèn biểu đồ: Tổng quan EDA]
![Prostate EDA](../../results/Prostate/eda/plot/countplot.png)

**Chú thích:**
- Mục đích: Kiểm tra xem bộ dữ liệu có bị mất cân bằng (imbalanced) hay không.
- Cách đọc: Trục hoành (V1) thể hiện các nhãn lớp (0 và 1), trục tung (count) là số lượng mẫu của từng lớp.

## 2) Tiền xử lý dữ liệu

- Điểm vào notebook:
- `notebook/Prostate/02_preprocess.ipynb`
- Quy ước thư mục đầu ra: `data/processed/Prostate/01_clean/`

## 3) Lọc đặc trưng (Filter Selection)

- Điểm vào notebook:
- `notebook/Prostate/03_filter_selection.ipynb`
- Tệp báo cáo: `results/Prostate/filter/reports/evaluation_Prostate.txt`

[Chèn biểu đồ: So sánh Filter Selection]
![Prostate Filter Selection](../../results/Prostate/filter/plots/evaluation_Prostate.png)

**Chú thích:**
- Mục đích: So sánh hiệu năng các phương pháp filter để chọn ra nhóm đặc trưng tốt nhất cho bước tiếp theo.
- Cách đọc: Trục hoành là các phương pháp filter, trục tung là điểm đánh giá; cột/điểm càng cao thì phương pháp càng tốt.

## 4) Mô hình hóa (so sánh ở giai đoạn filter)

- Điểm vào notebook:
- `notebook/Prostate/04_modeling.ipynb`
- Kết quả modeling được lưu dưới `results/Prostate/filter/` khi có sẵn.

## 5) Ensemble Filter (Bỏ phiếu + tập đặc trưng union)

- Điểm vào notebook:
- `notebook/Prostate/05_ensemble.ipynb`
- Tệp seed pool: `data/processed/Prostate/03_ensemble/top50_features_voting.csv`
- Kích thước seed pool: 10
- Đặc trưng có số phiếu cao nhất: `V1840(5)`, `V2620(4)`, `V5017(4)`, `V4702(4)`, `V3666(4)`

[Chèn biểu đồ: Bỏ phiếu Ensemble / Đặc trưng Union]
![Prostate Ensemble Voting](../../results/Prostate/ensemble/plots/top50_features_voting.png)

**Chú thích:**
- Mục đích: Hiển thị mức độ đồng thuận của các phương pháp filter khi bỏ phiếu chọn đặc trưng.
- Cách đọc: Trục hoành là tên đặc trưng, trục tung là số phiếu (vote count); đặc trưng có phiếu cao hơn được ưu tiên hơn.

## 6) Wrapper: Sklearn SFS (chạy Raw vs Union)

- Điểm vào script:
- `notebook/Prostate/06_sklearn_sfs-raw.py`
- `notebook/Prostate/06_sklearn_sfs-union.py`

| Biến thể | Sklearn Số đặc trưng chọn | Sklearn Global Best | Sklearn Thời gian fit (ms) |
|---|---:|---:|---:|
| Raw | 3 | 0.9614 | 321,108 |
| Union | 3 | 0.9514 | 10,968 |

## 7) Wrapper: Seeded SFS (chạy Raw vs Union)

- Điểm vào script:
- `notebook/Prostate/07_sfs-raw.py`
- `notebook/Prostate/07_sfs-union.py`

| Biến thể | Seeded Số đặc trưng chọn | Seeded Global Best | Seeded Thời gian fit (ms) |
|---|---:|---:|---:|
| Raw | 4 | 0.9714 | 63,907 |
| Union | 3 | 0.9614 | 3,566 |

## 8) Đánh giá Accuracy (so sánh Raw vs Union)

- Điểm vào notebook:
- `notebook/Prostate/8_accuracu_evaluate.ipynb`
- `notebook/Prostate/8_accuracu_evaluate_union.ipynb`

[Chèn biểu đồ: So sánh Accuracy Raw vs Union]
![Prostate Accuracy Evaluation](../../results/Prostate/evaluation/plots/wrapper_sfs_comparison_sk_raw_seeded_raw_Prostate.png)

**Chú thích:**
- Mục đích: So sánh độ chính xác giữa các cấu hình wrapper (Sklearn SFS và Seeded SFS) theo từng biến thể dữ liệu.
- Cách đọc:
  - Trục hoành là từng cấu hình/phương pháp, trục tung là accuracy; giá trị cao hơn thể hiện hiệu năng tốt hơn.
  - Vạch đen thẳng đứng (Error bar): Thể hiện độ lệch chuẩn (Standard Deviation) qua các fold cross-validation. Vạch này càng ngắn chứng tỏ mô hình dự đoán càng ổn định, ít biến động.
![Prostate Accuracy Evaluation](../../results/Prostate/evaluation/plots/wrapper_sfs_comparison_sk_union_seeded_union_Prostate.png)

**Chú thích:**
- Mục đích: So sánh độ chính xác giữa các cấu hình wrapper (Sklearn SFS và Seeded SFS) theo từng biến thể dữ liệu.
- Cách đọc:
  - Trục hoành là từng cấu hình/phương pháp, trục tung là accuracy; giá trị cao hơn thể hiện hiệu năng tốt hơn.
  - Vạch đen thẳng đứng (Error bar): Thể hiện độ lệch chuẩn (Standard Deviation) qua các fold cross-validation. Vạch này càng ngắn chứng tỏ mô hình dự đoán càng ổn định, ít biến động.

- **Quan sát:** Biến thể raw cho accuracy nhỉnh hơn, trong khi union nhanh hơn đáng kể.
- **Giải thích:** Union thu hẹp không gian đặc trưng và giảm tính toán, nhưng có thể bỏ sót tín hiệu chỉ có trong raw.
- **Kết luận:** Dùng raw khi cần accuracy cao nhất và dùng union cho chu kỳ phát triển nhanh hơn.

- Cấu hình tốt nhất (raw): `seeded + LogReg`, accuracy trung bình **0.9714**, std 0.0639
- Cấu hình tốt nhất (union): `seeded + LogReg`, accuracy trung bình 0.9614, std 0.0622
- Final selected features: 4 features (raw seeded run)

## 9) Đánh giá thời gian (so sánh thời gian fit Raw vs Union)

- Điểm vào notebook:
- `notebook/Prostate/9_time_evaluate.ipynb`
- `notebook/Prostate/9_time_evaluate_union.ipynb`

[Chèn biểu đồ: So sánh thời gian Raw vs Union]
![Prostate Time Evaluation](../../results/Prostate/evaluation/plots/time_comparison_raw_seeded_vs_raw_sklearn.png)

**Chú thích:**
- Mục đích: So sánh chi phí thời gian huấn luyện giữa các phương pháp wrapper trên cùng bộ dữ liệu.
- Cách đọc: Trục hoành là phương pháp/cấu hình, trục tung là tổng thời gian fit (ms); cột thấp hơn nghĩa là chạy nhanh hơn.
![Prostate Time Evaluation](../../results/Prostate/evaluation/plots/time_comparison_union_seeded_vs_union_sklearn.png)

**Chú thích:**
- Mục đích: So sánh chi phí thời gian huấn luyện giữa các phương pháp wrapper trên cùng bộ dữ liệu.
- Cách đọc: Trục hoành là phương pháp/cấu hình, trục tung là tổng thời gian fit (ms); cột thấp hơn nghĩa là chạy nhanh hơn.

- **Quan sát:** Các lần chạy union thường nhanh hơn raw trên hầu hết phương pháp wrapper.
- **Giải thích:** Union làm giảm không gian ứng viên, từ đó giảm tổng số lần fit mô hình.
- **Kết luận:** Dùng union để lặp thử nhanh; dùng raw khi cần tối đa hóa wrapper score.


## 10) Đánh Giá Cuối Cùng (So Sánh Tất Cả Phương Pháp)

- Điểm vào notebook:
- `notebook/Prostate/10_final_evaluate.ipynb`
- Báo cáo: `results/Prostate/evaluation/reports/final_evaluation_all_methods_prostate_Prostate.txt`

[Biểu Đồ: Đánh Giá Cuối Cùng - Tất Cả Phương Pháp]
![Prostate Final Evaluation](../../results/Prostate/evaluation/plots/final_evaluation_all_methods_prostate_Prostate.png)

**Chú Thích:**
- Mục đích: So sánh tất cả phương pháp lựa chọn đặc trưng (Filter, Ensemble, Sklearn SFS, Seeded SFS) với cả hai mô hình LogReg và Tree.
- Cách đọc:
  - Trục X liệt kê tất cả các kết hợp phương pháp/mô hình (ví dụ: "Sklearn_SFS_Raw + LogReg").
  - Trục Y hiển thị độ chính xác cross-validation; các cột cao hơn cho biết hiệu suất tốt hơn.
  - Các thanh lỗi dọc hiển thị độ lệch chuẩn (Std) trên các fold; các thanh ngắn hơn chỉ ra mô hình ổn định hơn.

| Xếp Hạng | Phương Pháp + Mô Hình | CV Fold | Accuracy Trung Bình | Std | Median | Min | Max |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 | Seeded_SFS_Raw + LogReg | 10 | 0.9714 | 0.0602 | 1.0000 | 0.8571 | 1.0000 |
| 2 | Seeded_SFS_Union + LogReg | 10 | 0.9614 | 0.0586 | 1.0000 | 0.8571 | 1.0000 |
| 3 | CHI_SQUARED + LogReg | 5 | 0.9514 | 0.0583 | 0.9500 | 0.8571 | 1.0000 |

**Quan Sát Chính:**
- Cấu hình tốt nhất: Seeded_SFS_Raw + LogReg với độ chính xác 0.9714 (σ=0.0602)
- Xếp thứ hai: Seeded_SFS_Union + LogReg với độ chính xác 0.9614
- Khuyến nghị: Xem so sánh chi tiết trong biểu đồ và tệp báo cáo ở trên.