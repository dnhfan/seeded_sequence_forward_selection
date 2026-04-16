# Pipeline Lọc Đặc Trưng (Filter Selection)

_Read in [English](01_filter_selection.md)._ 

**1. Tổng quan / Mục đích:**  
Pipeline này thực hiện bước giảm chiều đặc trưng ban đầu bằng các phương pháp lọc (filter-based). Mục tiêu là loại nhanh các đặc trưng yếu hoặc dư thừa trước khi chạy wrapper tốn chi phí, đồng thời vẫn giữ được tập ứng viên đa dạng cho bước ensemble voting và SFS phía sau.

**2. Đầu vào:**  
- File CSV dữ liệu gốc có cấu trúc: cột đầu tiên = nhãn mục tiêu (`y`), các cột còn lại = đặc trưng (`X`)  
  - Mẫu đường dẫn chuẩn: `data/raw/<dataset>.csv` (qua `ProjectPath.raw_path`)  
- Tham số cấu hình:
  - `data_name` (tên bộ dữ liệu)
  - `n_features` (số đặc trưng top-k giữ lại cho mỗi phương pháp, thường là 50)
  - `method` thuộc:
    - `variance`
    - `correlation`
    - `chi_squared`
    - `mutual_information`
    - `anova_f_test`
  - `random_state` (dùng cho MI)
- Mã nguồn chính:
  - `src/filter/filter_selection.py`
  - `src/filter/filter_algorithms.py`

**3. Logic thực thi từng bước:**  
1. **Khởi tạo bộ chọn (`FeatureFilter`)**  
   Pipeline tạo `FeatureFilter(method, n_features, random_state)`.  
   Class này kiểm tra method hợp lệ và chuẩn bị trạng thái nội bộ (`selected_features_`, `feature_scores_`).

2. **Tách dữ liệu thành `X` và `y`**  
   Nhãn mục tiêu được tách riêng vì nhiều phương pháp lọc là supervised (cần nhãn), trong khi variance là unsupervised.

3. **Chạy logic chấm điểm theo từng phương pháp**  
   Tùy `method`, pipeline gọi đúng thuật toán:
   - **Variance (`calc_variance`)**: tính phương sai từng đặc trưng và xếp hạng giảm dần.
   - **Correlation (`calc_correlation`)**:  
     a) tạo ma trận tương quan tuyệt đối giữa các đặc trưng,  
     b) loại cột tương quan quá cao (`|corr| > 0.95`),  
     c) xếp hạng phần còn lại bằng ANOVA F-score theo nhãn.  
     Cách này giúp tránh chọn quá nhiều đặc trưng gần như trùng thông tin.
   - **Chi-squared (`calc_chi_squared`)**:  
     a) kiểm tra giá trị âm,  
     b) nếu có âm thì chuẩn hóa `MinMaxScaler` về miền không âm,  
     c) tính điểm chi2 bằng `SelectKBest`.  
     Bước này đảm bảo đúng giả định của chi-squared.
   - **Mutual Information (`calc_mutual_info`)**: ước lượng mức phụ thuộc phi tuyến giữa từng đặc trưng và nhãn bằng `mutual_info_classif`.
   - **ANOVA F-test (`calc_anova`)**: tính F-statistics với `SelectKBest(f_classif)`.

4. **Chọn top-k đặc trưng**  
   Với mỗi method, đặc trưng được xếp theo score và giữ lại top `n_features`.  
   Nếu `n_features` yêu cầu lớn hơn số cột thực tế, pipeline tự giới hạn an toàn.

5. **Biến đổi dữ liệu**  
   `transform()` trả về `X` chỉ gồm các cột đã chọn.  
   `fit_transform()` chạy fit + transform trong một lần gọi.

6. **Lưu CSV sau lọc**  
   `save_filtered_data()` ghép lại `[target + selected features]` rồi lưu vào:  
   `data/processed/<dataset>/02_filter/<dataset>_<method>_<n_features>features.csv`

**4. Đầu ra / Artifacts:**  
- Một file CSV sau lọc cho mỗi method trong `02_filter/`:
  - Ví dụ: `data/processed/colon1/02_filter/colon1_anova_f_test_50features.csv`
- Trạng thái trong bộ nhớ sau khi fit:
  - `selected_features_` (danh sách đặc trưng đã chọn theo thứ tự)
  - `feature_scores_` (map đặc trưng -> điểm)
  - scaler tùy chọn cho nhánh chi2
- Các file sau lọc này là đầu vào cho:
  - Ensemble voting (`03_ensemble`)
  - Tạo Union features
  - Đánh giá baseline cho nhóm filter methods
