# Pipeline Wrapper - Sklearn Sequential Feature Selector (SFS)

_Read in [English](03_wrapper_sklearn_sfs.md)._ 

**1. Tổng quan / Mục đích:**  
Pipeline này chạy Sklearn Sequential Feature Selector chuẩn như một baseline wrapper tham chiếu. Mục tiêu là đo hiệu năng của forward SFS chuẩn (độ chính xác + thời gian fit) trên cả hai biến thể Raw và Union, rồi lưu artifacts theo cùng chuẩn để so sánh công bằng với Seeded SFS.

**2. Đầu vào:**  
- DataFrame đầu vào có format: cột đầu là target, các cột còn lại là features.
- Biến thể dữ liệu:
  - **Raw**: đọc từ `data/raw/<dataset>.csv`
  - **Union**: tạo bằng `create_union_features(...)` từ output filter
- Script chạy (ví dụ theo từng dataset):
  - `notebook/<dataset>/06_sklearn_sfs-raw.py`
  - `notebook/<dataset>/06_sklearn_sfs-union.py`
- Class/file chính:
  - `src/wrapper/sklearn_sfs.py` (`SklearnSFSSelector`)
  - `src/wrapper/base.py` (`BaseWrapperSelector`)
  - `src/wrapper/models.py` (`get_model`)
  - `src/utils/experiment_paths.py` (chuẩn thư mục run)

**3. Logic thực thi từng bước:**  
1. **Khởi tạo wrapper selector**  
   Notebook tạo `SklearnSFSSelector(...)` với:
   - `data_name`, `n_features`
   - `dataset_variant` (`raw` hoặc `union`)
   - `voting_csv_name` (giữ để đồng nhất interface)
   - cấu hình timer (`using_timer=True`, `unit="ms"`)

2. **Chuẩn bị dữ liệu**  
   `BaseWrapperSelector.run_sfs()` tách DataFrame:
   - `y_in = df.iloc[:, 0]`
   - `X_in = df.iloc[:, 1:]`  
   Bước này thống nhất contract dữ liệu cho mọi wrapper.

3. **Tạo cấu hình chạy SFS**  
   Tham số chính gồm:
   - `max_features` (`"auto"` hoặc số nguyên)
   - `cv`
   - `model` (map qua `get_model`)
   - `scoring` (thường là accuracy)
   - `direction` (mặc định forward)

4. **Khởi tạo và fit sklearn SFS**  
   Trong `_execute_core()`:
   - tạo `StratifiedKFold(shuffle=True, random_state=42)` để deterministic,
   - tạo `SequentialFeatureSelector` theo model/scoring/cv đã cấu hình,
   - chạy fit trong `TimerContext` để đo tổng thời gian fit.

5. **Lấy đặc trưng đã chọn và dựng dataset cuối**  
   Sau fit:
   - lấy support mask từ selector,
   - tạo `X_selected_df`,
   - ghép target + selected features thành DataFrame kết quả.

6. **Tính lại global best score trên tập đã chọn**  
   Vì sklearn SFS không trả full iteration history, pipeline chủ động chạy `cross_val_score` trên tập đặc trưng cuối để lấy `global_best_score` (CV mean).

7. **Ghi history dạng tóm tắt**  
   Tạo báo cáo text gồm:
   - cấu hình
   - tổng fit time
   - best score cuối
   - danh sách features đã chọn  
   Đồng thời nêu rõ sklearn SFS không có log chi tiết từng vòng lặp.

8. **Lưu artifacts theo chuẩn base wrapper**  
   `BaseWrapperSelector._save_sfs_output()` ghi:
   - CSV wrapper output trong `data/processed/.../04_wrapper/...`
   - run artifacts trong `results/<dataset>/wrapper/<variant>/sklearnsfsselector/run_<timestamp>/...`
   - metrics gồm `total_fit_time_ms`

**4. Đầu ra / Artifacts:**  
- Dataset sau chọn đặc trưng:
  - `data/processed/<dataset>/04_wrapper/<variant>/sklearnsfsselector/<dataset>_sklearnsfsselector_*.csv`
- Cấu trúc run artifacts:
  - `results/<dataset>/wrapper/<variant>/sklearnsfsselector/run_YYYYMMDD_HHMMSS/`
    - `history/history.txt`
    - `features/selected_features.csv`
    - `metrics/metrics.json`
    - `metrics/metrics.csv` (có `total_fit_time_ms`)
- Metrics được lưu:
  - `n_features_selected`
  - `global_best_score`
  - `total_fit_time_ms`
  - metadata dataset + variant
