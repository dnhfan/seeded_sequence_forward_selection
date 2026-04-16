# NCI Kết quả và Đánh giá

_Đọc bản tiếng Anh tại [result-nci.md](result-nci.md)_

[Quay lại mục lục](./README.vi.md)

## 1) EDA (Phân tích khám phá dữ liệu)

- Điểm vào notebook:
- `notebook/NCI/01_eda.ipynb`

[Chèn biểu đồ: Tổng quan EDA]
![NCI EDA](../../results/NCI/eda/plot/countplot.png)

**Chú thích:**
- Mục đích: Kiểm tra xem bộ dữ liệu có bị mất cân bằng (imbalanced) hay không.
- Cách đọc: Trục hoành (V1) thể hiện các nhãn lớp (0 và 1), trục tung (count) là số lượng mẫu của từng lớp.

## 2) Tiền xử lý dữ liệu

- Điểm vào notebook:
- Không có rõ ràng trong thư mục notebook hiện tại.
- Quy ước thư mục đầu ra: `data/processed/NCI/01_clean/`

## 3) Lọc đặc trưng (Filter Selection)

- Điểm vào notebook:
- `notebook/NCI/02_filter_selection.ipynb`
- Tệp báo cáo: `results/NCI/filter/reports/evaluation_NCI.txt`

[Chèn biểu đồ: So sánh Filter Selection]
![NCI Filter Selection](../../results/NCI/filter/plots/evaluation_NCI.png)

**Chú thích:**
- Mục đích: So sánh hiệu năng các phương pháp filter để chọn ra nhóm đặc trưng tốt nhất cho bước tiếp theo.
- Cách đọc: Trục hoành là các phương pháp filter, trục tung là điểm đánh giá; cột/điểm càng cao thì phương pháp càng tốt.

## 4) Mô hình hóa (so sánh ở giai đoạn filter)

- Điểm vào notebook:
- `notebook/NCI/03_modeling.ipynb`
- Kết quả modeling được lưu dưới `results/NCI/filter/` khi có sẵn.

## 5) Ensemble Filter (Bỏ phiếu + tập đặc trưng union)

- Điểm vào notebook:
- `notebook/NCI/04_ensemble.ipynb`
- Tệp seed pool: `data/processed/NCI/03_ensemble/top50_features_voting.csv`
- Kích thước seed pool: 10
- Đặc trưng có số phiếu cao nhất: `V155(4)`, `V4968(4)`, `V4516(4)`, `V4618(4)`, `V4581(4)`

[Chèn biểu đồ: Bỏ phiếu Ensemble / Đặc trưng Union]
![NCI Ensemble Voting](../../results/NCI/ensemble/plots/top50_features_voting.png)

**Chú thích:**
- Mục đích: Hiển thị mức độ đồng thuận của các phương pháp filter khi bỏ phiếu chọn đặc trưng.
- Cách đọc: Trục hoành là tên đặc trưng, trục tung là số phiếu (vote count); đặc trưng có phiếu cao hơn được ưu tiên hơn.

## 6) Wrapper: Sklearn SFS (chạy Raw vs Union)

- Điểm vào script:
- `notebook/NCI/06_sklearn_sfs-raw.py`
- `notebook/NCI/06_sklearn_sfs-union.py`

| Biến thể | Sklearn Số đặc trưng chọn | Sklearn Global Best | Sklearn Thời gian fit (ms) |
|---|---:|---:|---:|
| Raw | 9 | 0.8551 | 825,324 |
| Union | 8 | 0.8192 | 29,053 |

## 7) Wrapper: Seeded SFS (chạy Raw vs Union)

- Điểm vào script:
- `notebook/NCI/07_sfs-raw.py`
- `notebook/NCI/07_sfs-union.py`

| Biến thể | Seeded Số đặc trưng chọn | Seeded Global Best | Seeded Thời gian fit (ms) |
|---|---:|---:|---:|
| Raw | 7 | 0.8833 | 223,827 |
| Union | 6 | 0.8359 | 7,373 |

## 8) Đánh giá Accuracy (so sánh Raw vs Union)

- Điểm vào notebook:
- `notebook/NCI/8_accuracu_evaluate.ipynb`
- `notebook/NCI/8_accuracu_evaluate_union.ipynb`

[Chèn biểu đồ: So sánh Accuracy Raw vs Union]
![NCI Accuracy Evaluation](../../results/NCI/evaluation/plots/wrapper_sfs_comparison_sk_raw_seeded_raw_NCI.png)

**Chú thích:**
- Mục đích: So sánh độ chính xác giữa các cấu hình wrapper (Sklearn SFS và Seeded SFS) theo từng biến thể dữ liệu.
- Cách đọc:
  - Trục hoành là từng cấu hình/phương pháp, trục tung là accuracy; giá trị cao hơn thể hiện hiệu năng tốt hơn.
  - Vạch đen thẳng đứng (Error bar): Thể hiện độ lệch chuẩn (Standard Deviation) qua các fold cross-validation. Vạch này càng ngắn chứng tỏ mô hình dự đoán càng ổn định, ít biến động.
![NCI Accuracy Evaluation](../../results/NCI/evaluation/plots/wrapper_sfs_comparison_sk_union_seeded_union_NCI.png)

**Chú thích:**
- Mục đích: So sánh độ chính xác giữa các cấu hình wrapper (Sklearn SFS và Seeded SFS) theo từng biến thể dữ liệu.
- Cách đọc:
  - Trục hoành là từng cấu hình/phương pháp, trục tung là accuracy; giá trị cao hơn thể hiện hiệu năng tốt hơn.
  - Vạch đen thẳng đứng (Error bar): Thể hiện độ lệch chuẩn (Standard Deviation) qua các fold cross-validation. Vạch này càng ngắn chứng tỏ mô hình dự đoán càng ổn định, ít biến động.

- **Quan sát:** Biến thể raw vượt union ở cả wrapper score và điểm đánh giá.
- **Giải thích:** Các đặc trưng raw bổ sung có vẻ giữ lại tín hiệu phân biệt hữu ích.
- **Kết luận:** Giữ biến thể raw làm mặc định cho các lần benchmark NCI.

- Cấu hình tốt nhất (raw): `seeded + LogReg`, accuracy trung bình **0.7872**, std 0.0937
- Cấu hình tốt nhất (union): `sklearn + LogReg`, accuracy trung bình 0.7385, std 0.1053

## 9) Đánh giá thời gian (so sánh thời gian fit Raw vs Union)

- Điểm vào notebook:
- `notebook/NCI/9_time_evaluate.ipynb`
- `notebook/NCI/9_time_evaluate_union.ipynb`

[Chèn biểu đồ: So sánh thời gian Raw vs Union]
![NCI Time Evaluation](../../results/NCI/evaluation/plots/time_comparison_raw_seeded_vs_raw_sklearn.png)

**Chú thích:**
- Mục đích: So sánh chi phí thời gian huấn luyện giữa các phương pháp wrapper trên cùng bộ dữ liệu.
- Cách đọc: Trục hoành là phương pháp/cấu hình, trục tung là tổng thời gian fit (ms); cột thấp hơn nghĩa là chạy nhanh hơn.
![NCI Time Evaluation](../../results/NCI/evaluation/plots/time_comparison_union_seeded_vs_union_sklearn.png)

**Chú thích:**
- Mục đích: So sánh chi phí thời gian huấn luyện giữa các phương pháp wrapper trên cùng bộ dữ liệu.
- Cách đọc: Trục hoành là phương pháp/cấu hình, trục tung là tổng thời gian fit (ms); cột thấp hơn nghĩa là chạy nhanh hơn.

- **Quan sát:** Các lần chạy union thường nhanh hơn raw trên hầu hết phương pháp wrapper.
- **Giải thích:** Union làm giảm không gian ứng viên, từ đó giảm tổng số lần fit mô hình.
- **Kết luận:** Dùng union để lặp thử nhanh; dùng raw khi cần tối đa hóa wrapper score.
