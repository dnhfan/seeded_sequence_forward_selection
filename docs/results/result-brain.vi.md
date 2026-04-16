# Brain Kết quả và Đánh giá

_Đọc bản tiếng Anh tại [result-brain.md](result-brain.md)_

[Quay lại mục lục](./README.vi.md)

## 1) EDA (Phân tích khám phá dữ liệu)

- Điểm vào notebook:
- `notebook/Brain/01_eda.ipynb`

[Chèn biểu đồ: Tổng quan EDA]
![Brain EDA](../../results/Brain/eda/plot/countplot.png)

**Chú thích:**

- Mục đích: Kiểm tra xem bộ dữ liệu có bị mất cân bằng (imbalanced) hay không.
- Cách đọc: Trục hoành (V1) thể hiện các nhãn lớp (0 và 1), trục tung (count) là số lượng mẫu của từng lớp.

## 2) Tiền xử lý dữ liệu

- Điểm vào notebook:
- Không có rõ ràng trong thư mục notebook hiện tại.
- Quy ước thư mục đầu ra: `data/processed/Brain/01_clean/`

## 3) Lọc đặc trưng (Filter Selection)

- Điểm vào notebook:
- `notebook/Brain/02_Filter_selection.ipynb`
- Tệp báo cáo: `results/Brain/filter/reports/filter_compare_50features_Brain.txt`

[Chèn biểu đồ: So sánh Filter Selection]
![Brain Filter Selection](../../results/Brain/filter/plots/filter_compare_50features_Brain.png)

**Chú thích:**

- Mục đích: So sánh hiệu năng các phương pháp filter để chọn ra nhóm đặc trưng tốt nhất cho bước tiếp theo.
- Cách đọc: Trục hoành là các phương pháp filter, trục tung là điểm đánh giá; cột/điểm càng cao thì phương pháp càng tốt.

## 4) Mô hình hóa (so sánh ở giai đoạn filter)

- Điểm vào notebook:
- `notebook/Brain/03_Modeling.ipynb`
- Kết quả modeling được lưu dưới `results/Brain/filter/` khi có sẵn.

## 5) Ensemble Filter (Bỏ phiếu + tập đặc trưng union)

- Điểm vào notebook:
- `notebook/Brain/04_Ensemble_fitler_selection.ipynb`
- `notebook/Brain/05_Union.ipynb`
- Tệp seed pool: `data/processed/Brain/03_ensemble/top50_features_voting.csv`
- Kích thước seed pool: 10
- Đặc trưng có số phiếu cao nhất: `V1893(5)`, `V523(4)`, `V541(4)`, `V1050(3)`, `V2332(3)`

[Chèn biểu đồ: Bỏ phiếu Ensemble / Đặc trưng Union]
![Brain Ensemble Voting](../../results/Brain/ensemble/plot/top50_features_voting.png)

**Chú thích:**

- Mục đích: Hiển thị mức độ đồng thuận của các phương pháp filter khi bỏ phiếu chọn đặc trưng.
- Cách đọc: Trục hoành là tên đặc trưng, trục tung là số phiếu (vote count); đặc trưng có phiếu cao hơn được ưu tiên hơn.

## 6) Wrapper: Sklearn SFS (chạy Raw vs Union)

- Điểm vào script:
- `notebook/Brain/06_sklearn_sfs-raw.py`
- `notebook/Brain/06_sklearn_sfs-union.py`

| Biến thể | Sklearn Số đặc trưng chọn | Sklearn Global Best | Sklearn Thời gian fit (ms) |
| -------- | ------------------------: | ------------------: | -------------------------: |
| Raw      |                         6 |              0.9295 |                    732,895 |
| Union    |                         4 |              0.8795 |                     13,169 |

## 7) Wrapper: Seeded SFS (chạy Raw vs Union)

- Điểm vào script:
- `notebook/Brain/07_sfs-raw.py`
- `notebook/Brain/07_sfs-union.py`

| Biến thể | Seeded Số đặc trưng chọn | Seeded Global Best | Seeded Thời gian fit (ms) |
| -------- | -----------------------: | -----------------: | ------------------------: |
| Raw      |                       11 |             0.9295 |                   112,591 |
| Union    |                        6 |                0.9 |                     3,555 |

## 8) Đánh giá Accuracy (so sánh Raw vs Union)

- Điểm vào notebook:
- `notebook/Brain/08_accuracy_evaluate.ipynb`
- `notebook/Brain/08_accuracy_evaluate_union.ipynb`

[Chèn biểu đồ: So sánh Accuracy Raw vs Union]
![Brain Accuracy Evaluation](../../results/Brain/evaluation/plots/evaluation_Brain.png)

**Chú thích:**

- Mục đích: So sánh độ chính xác giữa các cấu hình wrapper (Sklearn SFS và Seeded SFS) theo từng biến thể dữ liệu.
- Cách đọc:
  - Trục hoành là từng cấu hình/phương pháp, trục tung là accuracy; giá trị cao hơn thể hiện hiệu năng tốt hơn.
  - Vạch đen thẳng đứng (Error bar): Thể hiện độ lệch chuẩn (Standard Deviation) qua các fold cross-validation. Vạch này càng ngắn chứng tỏ mô hình dự đoán càng ổn định, ít biến động.
![Brain Accuracy Evaluation](../../results/Brain/evaluation/plots/wrapper_sfs_comparison_union_Brain.png)

**Chú thích:**

- Mục đích: So sánh độ chính xác giữa các cấu hình wrapper (Sklearn SFS và Seeded SFS) theo từng biến thể dữ liệu.
- Cách đọc:
  - Trục hoành là từng cấu hình/phương pháp, trục tung là accuracy; giá trị cao hơn thể hiện hiệu năng tốt hơn.
  - Vạch đen thẳng đứng (Error bar): Thể hiện độ lệch chuẩn (Standard Deviation) qua các fold cross-validation. Vạch này càng ngắn chứng tỏ mô hình dự đoán càng ổn định, ít biến động.

## 9) Đánh giá thời gian (so sánh thời gian fit Raw vs Union)

- Điểm vào notebook:
- `notebook/Brain/9_time_evaluate.ipynb`
- `notebook/Brain/9_time_evaluate_union.ipynb`

[Chèn biểu đồ: So sánh thời gian Raw vs Union]
![Brain Time Evaluation](../../results/Brain/evaluation/plots/time_comparison_seeded3_vs_sklearn_brain.png)

**Chú thích:**

- Mục đích: So sánh chi phí thời gian huấn luyện giữa các phương pháp wrapper trên cùng bộ dữ liệu.
- Cách đọc: Trục hoành là phương pháp/cấu hình, trục tung là tổng thời gian fit (ms); cột thấp hơn nghĩa là chạy nhanh hơn.
  ![Brain Time Evaluation](../../results/Brain/evaluation/plots/time_comparison_union_seeded_vs_union_sklearn.png)

**Chú thích:**

- Mục đích: So sánh chi phí thời gian huấn luyện giữa các phương pháp wrapper trên cùng bộ dữ liệu.
- Cách đọc: Trục hoành là phương pháp/cấu hình, trục tung là tổng thời gian fit (ms); cột thấp hơn nghĩa là chạy nhanh hơn.

- **Quan sát:** Các lần chạy union thường nhanh hơn raw trên hầu hết phương pháp wrapper.
- **Giải thích:** Union làm giảm không gian ứng viên, từ đó giảm tổng số lần fit mô hình.
- **Kết luận:** Dùng union để lặp thử nhanh; dùng raw khi cần tối đa hóa wrapper score.
