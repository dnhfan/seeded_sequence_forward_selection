# Breast2classes Kết quả và Đánh giá

_Đọc bản tiếng Anh tại [result-breast2classes.md](result-breast2classes.md)_

[Quay lại mục lục](./README.vi.md)

## 1) EDA (Phân tích khám phá dữ liệu)

- Điểm vào notebook:
- `notebook/Breast2classes/01_eda.ipynb`

[Chèn biểu đồ: Tổng quan EDA]
![Breast2classes EDA](../../results/Breast2classes/eda/plot/countplot.png)

**Chú thích:**
- Mục đích: Kiểm tra xem bộ dữ liệu có bị mất cân bằng (imbalanced) hay không.
- Cách đọc: Trục hoành (V1) thể hiện các nhãn lớp (0 và 1), trục tung (count) là số lượng mẫu của từng lớp.

## 2) Tiền xử lý dữ liệu

- Điểm vào notebook:
- `notebook/Breast2classes/02_preprocess.ipynb`
- Quy ước thư mục đầu ra: `data/processed/Breast2classes/01_clean/`

## 3) Lọc đặc trưng (Filter Selection)

- Điểm vào notebook:
- `notebook/Breast2classes/03_filter_selection.ipynb`
- Tệp báo cáo: `results/Breast2classes/filter/reports/filter_compare_50features_Breast2classes.txt`

[Chèn biểu đồ: So sánh Filter Selection]
![Breast2classes Filter Selection](../../results/Breast2classes/filter/plots/filter_compare_50features_Breast2classes.png)

**Chú thích:**
- Mục đích: So sánh hiệu năng các phương pháp filter để chọn ra nhóm đặc trưng tốt nhất cho bước tiếp theo.
- Cách đọc: Trục hoành là các phương pháp filter, trục tung là điểm đánh giá; cột/điểm càng cao thì phương pháp càng tốt.
![Breast2classes Filter Selection](../../results/Breast2classes/filter/plots/model_comparison_top50_2026-04-09.png)

**Chú thích:**
- Mục đích: So sánh hiệu năng các phương pháp filter để chọn ra nhóm đặc trưng tốt nhất cho bước tiếp theo.
- Cách đọc: Trục hoành là các phương pháp filter, trục tung là điểm đánh giá; cột/điểm càng cao thì phương pháp càng tốt.

## 4) Mô hình hóa (so sánh ở giai đoạn filter)

- Điểm vào notebook:
- `notebook/Breast2classes/04_modeling.ipynb`
- Kết quả modeling được lưu dưới `results/Breast2classes/filter/` khi có sẵn.

## 5) Ensemble Filter (Bỏ phiếu + tập đặc trưng union)

- Điểm vào notebook:
- `notebook/Breast2classes/05_esemble_filter.ipynb`
- Tệp seed pool: `data/processed/Breast2classes/03_ensemble/top50_features_voting.csv`
- Kích thước seed pool: 10
- Đặc trưng có số phiếu cao nhất: `V2221(4)`, `V3818(4)`, `V1325(4)`, `V2273(4)`, `V4360(4)`

[Chèn biểu đồ: Bỏ phiếu Ensemble / Đặc trưng Union]
![Breast2classes Ensemble Voting](../../results/Breast2classes/ensemble/plots/top50_features_voting_2026-04-10.png)

**Chú thích:**
- Mục đích: Hiển thị mức độ đồng thuận của các phương pháp filter khi bỏ phiếu chọn đặc trưng.
- Cách đọc: Trục hoành là tên đặc trưng, trục tung là số phiếu (vote count); đặc trưng có phiếu cao hơn được ưu tiên hơn.

## 6) Wrapper: Sklearn SFS (chạy Raw vs Union)

- Điểm vào script:
- `notebook/Breast2classes/06_sklearn_sfs-raw.py`
- `notebook/Breast2classes/06_sklearn_sfs-union.py`

| Biến thể | Sklearn Số đặc trưng chọn | Sklearn Global Best | Sklearn Thời gian fit (ms) |
|---|---:|---:|---:|
| Raw | 4 | 0.8583 | 315,016 |
| Union | 4 | 0.8567 | 13,638 |

## 7) Wrapper: Seeded SFS (chạy Raw vs Union)

- Điểm vào script:
- `notebook/Breast2classes/07_sfs-raw.py`
- `notebook/Breast2classes/07_sfs-union.py`

| Biến thể | Seeded Số đặc trưng chọn | Seeded Global Best | Seeded Thời gian fit (ms) |
|---|---:|---:|---:|
| Raw | 11 | 0.935 | 102,179 |
| Union | 9 | 0.8958 | 5,660 |

## 8) Đánh giá Accuracy (so sánh Raw vs Union)

- Điểm vào notebook:
- `notebook/Breast2classes/8_accuracu_evaluate.ipynb`
- `notebook/Breast2classes/8_accuracu_evaluate_union.ipynb`

[Chèn biểu đồ: So sánh Accuracy Raw vs Union]
![Breast2classes Accuracy Evaluation](../../results/Breast2classes/evaluation/plots/wrapper_sfs_comparison_sk_raw_seeded_raw_Breast2classes.png)

**Chú thích:**
- Mục đích: So sánh độ chính xác giữa các cấu hình wrapper (Sklearn SFS và Seeded SFS) theo từng biến thể dữ liệu.
- Cách đọc:
  - Trục hoành là từng cấu hình/phương pháp, trục tung là accuracy; giá trị cao hơn thể hiện hiệu năng tốt hơn.
  - Vạch đen thẳng đứng (Error bar): Thể hiện độ lệch chuẩn (Standard Deviation) qua các fold cross-validation. Vạch này càng ngắn chứng tỏ mô hình dự đoán càng ổn định, ít biến động.
![Breast2classes Accuracy Evaluation](../../results/Breast2classes/evaluation/plots/wrapper_sfs_comparison_sk_union_seeded_union_Breast2classes.png)

**Chú thích:**
- Mục đích: So sánh độ chính xác giữa các cấu hình wrapper (Sklearn SFS và Seeded SFS) theo từng biến thể dữ liệu.
- Cách đọc:
  - Trục hoành là từng cấu hình/phương pháp, trục tung là accuracy; giá trị cao hơn thể hiện hiệu năng tốt hơn.
  - Vạch đen thẳng đứng (Error bar): Thể hiện độ lệch chuẩn (Standard Deviation) qua các fold cross-validation. Vạch này càng ngắn chứng tỏ mô hình dự đoán càng ổn định, ít biến động.

- **Quan sát:** Raw seeded cho kết quả tốt hơn đáng kể so với union seeded trong đánh giá.
- **Giải thích:** Thông tin phân biệt có vẻ trải rộng ra ngoài nhóm ứng viên bị giới hạn bởi union.
- **Kết luận:** Giữ raw seeded làm mặc định để đạt hiệu năng dự đoán tốt nhất.

- Cấu hình tốt nhất (raw): `seeded + LogReg`, accuracy trung bình **0.9083**, std 0.0381
- Cấu hình tốt nhất (union): `seeded + LogReg`, accuracy trung bình 0.8200, std 0.1188
- Final selected features (winning setup, raw seeded): 11 features

## 9) Đánh giá thời gian (so sánh thời gian fit Raw vs Union)

- Điểm vào notebook:
- `notebook/Breast2classes/9_time_evaluate.ipynb`
- `notebook/Breast2classes/9_time_evaluate_union.ipynb`

[Chèn biểu đồ: So sánh thời gian Raw vs Union]
![Breast2classes Time Evaluation](../../results/Breast2classes/evaluation/plots/time_comparison_raw_seeded_vs_raw_sklearn.png)

**Chú thích:**
- Mục đích: So sánh chi phí thời gian huấn luyện giữa các phương pháp wrapper trên cùng bộ dữ liệu.
- Cách đọc: Trục hoành là phương pháp/cấu hình, trục tung là tổng thời gian fit (ms); cột thấp hơn nghĩa là chạy nhanh hơn.
![Breast2classes Time Evaluation](../../results/Breast2classes/evaluation/plots/time_comparison_union_seeded_vs_union_sklearn.png)

**Chú thích:**
- Mục đích: So sánh chi phí thời gian huấn luyện giữa các phương pháp wrapper trên cùng bộ dữ liệu.
- Cách đọc: Trục hoành là phương pháp/cấu hình, trục tung là tổng thời gian fit (ms); cột thấp hơn nghĩa là chạy nhanh hơn.

- **Quan sát:** Các lần chạy union thường nhanh hơn raw trên hầu hết phương pháp wrapper.
- **Giải thích:** Union làm giảm không gian ứng viên, từ đó giảm tổng số lần fit mô hình.
- **Kết luận:** Dùng union để lặp thử nhanh; dùng raw khi cần tối đa hóa wrapper score.
