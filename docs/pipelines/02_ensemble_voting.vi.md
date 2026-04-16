# Pipeline Bỏ Phiếu Tổ Hợp (Ensemble Voting)

_Read in [English](02_ensemble_voting.md)._ 

**1. Tổng quan / Mục đích:**  
Pipeline này tổng hợp đặc trưng đã chọn từ nhiều filter methods và áp dụng cơ chế đếm phiếu để tìm các đặc trưng "hạt giống" (seed) ổn định. Mục tiêu là tạo bảng xếp hạng đồng thuận làm điểm khởi tạo chất lượng cho Seeded SFS.

**2. Đầu vào:**  
- Các file CSV sau lọc từ bước `02_filter`, mỗi method một file:
  - `data/processed/<dataset>/02_filter/<dataset>_<method>_<n_features>features.csv`
- Tham số:
  - `data_name`
  - `valid_methods` (thường 5 phương pháp)
  - `n_features` (dùng cho naming input/output)
  - `data_dir` (thư mục chứa filtered CSV)
- Mã nguồn chính:
  - `src/filter/ensemble.py` (`EnsembleFeatureSelector`)

**3. Logic thực thi từng bước:**  
1. **Khởi tạo bộ chọn ensemble**  
   `EnsembleFeatureSelector(data_name, valid_methods, n_features, data_dir)` dựng đường dẫn bằng `ProjectPath` và tạo sẵn thư mục output:
   - report dir (`results/.../ensemble/reports`)
   - plot dir (`results/.../ensemble/plots`)
   - csv dir (`data/processed/.../03_ensemble`)

2. **Thu thập danh sách cột đã chọn từ từng method**  
   Trong `run_voting()`, với mỗi method:
   - chỉ đọc header CSV (`nrows=0`) để nhẹ,
   - bỏ cột đầu tiên (target),
   - lấy tên các cột đặc trưng còn lại.

3. **Đếm phiếu giữa các method**  
   Tất cả đặc trưng được gộp lại và đếm bằng `Counter`.  
   Đặc trưng xuất hiện ở càng nhiều method thì số phiếu càng cao, thể hiện tính ổn định liên-phương-pháp.

4. **Tạo bảng xếp hạng phiếu**  
   Tạo DataFrame gồm cột:
   - `Feature`
   - `Votes`  
   Sau đó sắp xếp giảm dần theo `Votes`.

5. **Sinh artifacts (`generate_report_and_plot`)**  
   - Lấy top-N dòng để trực quan và báo cáo (`top_n_plot`).
   - Lưu biểu đồ cột số phiếu.
   - Lưu báo cáo text.
   - Lưu CSV xếp hạng đặc trưng top vote.

6. **Dùng output voting làm đầu vào cho Seeded SFS**  
   Seeded SFS đọc cột `Feature` từ file ranking này và lấy `n_seeds` dòng đầu làm tập seed khởi tạo (`load_seed_from_csv`).

**4. Đầu ra / Artifacts:**  
- CSV xếp hạng seed dùng cho máy:
  - `data/processed/<dataset>/03_ensemble/top<n_features>_features_voting.csv`
- Artifacts đọc cho người:
  - `results/<dataset>/ensemble/plots/top<n_features>_features_voting.png`
  - `results/<dataset>/ensemble/reports/top<n_features>_features_voting.txt`
- DataFrame xếp hạng trong bộ nhớ (`df_counts`) cho các bước kế tiếp.
