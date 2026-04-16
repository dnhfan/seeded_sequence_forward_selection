# Pipeline Đánh Giá Metrics (Accuracy + Fit Time, Raw vs Union)

_Read in [English](05_evaluation_metrics.md)._ 

**1. Tổng quan / Mục đích:**  
Pipeline này đánh giá và so sánh kết quả chọn đặc trưng theo nhiều phương pháp, mô hình và biến thể dữ liệu (Raw vs Union). Pipeline báo cáo cả hiệu năng dự đoán (thống kê accuracy) và hiệu năng tính toán (fit time), từ đó so sánh công bằng giữa Seeded SFS và Sklearn SFS.

**2. Đầu vào:**  
- Engine đánh giá:
  - `src/modeling/evaluation.py` (`ModelEvaluator`)
- Đầu vào cho đánh giá accuracy:
  - Bất kỳ file CSV nào có cột đầu là target và các cột sau là features
  - Nguồn dữ liệu thường dùng:
    - output filter (`02_filter`)
    - output wrapper (`04_wrapper`)
    - file bất kỳ truyền qua `evaluate_custom_file(...)`
- Đầu vào cho đánh giá thời gian:
  - `metrics/metrics.csv` từ run folder của wrapper, bắt buộc có cột `total_fit_time_ms`
- Các chiều so sánh phổ biến:
  - nhãn method (ví dụ `Sklearn_SFS_Raw`, `Seeded_SFS_Union`)
  - mô hình: Logistic Regression (`LogReg`) và Decision Tree (`Tree`)
  - biến thể: Raw và Union

**3. Logic thực thi từng bước:**  
1. **Khởi tạo evaluator**  
   `ModelEvaluator(data_name, n_features, max_iter, use_scaler, custom_base_dir)` chuẩn bị thư mục output:
   - reports dir
   - plots dir  
   Nếu truyền `custom_base_dir`, toàn bộ artifacts sẽ được lưu vào thư mục đó (thường dùng cho notebook so sánh wrapper).

2. **Nạp dữ liệu theo một chuẩn thống nhất**  
   `_load_data(file_path)` luôn tách:
   - `y = cột đầu`
   - `X = các cột còn lại`

3. **Huấn luyện và đánh giá bằng cross-validation (`_train_and_evaluate`)**  
   Với mỗi method label:
   - tạo `StratifiedKFold(shuffle=True, random_state=42)` để reproducible,
   - định nghĩa 2 mô hình:
     - Logistic Regression (có thể bọc thêm `StandardScaler`)
     - Decision Tree (`max_depth=5`)
   - chạy `cross_validate(..., scoring="accuracy")`,
   - lưu accuracy theo từng fold cho từng model.

4. **Đánh giá theo nhóm nguồn hoặc file chỉ định**  
   - `evaluate_filtered_features(data_dir)`: tự lặp qua các filter methods.
   - `evaluate_baseline(raw_path)`: đánh giá baseline toàn bộ đặc trưng gốc.
   - `evaluate_custom_file(file_path, method_label)`: đánh giá output wrapper hoặc file tùy ý.

5. **Sinh báo cáo accuracy + biểu đồ (`generate_report_and_plot`)**  
   - Tạo DataFrame ở mức fold từ toàn bộ kết quả tích lũy.
   - Vẽ bar chart (`Method` x `Model` kèm nhãn accuracy).
   - Tính thống kê tổng hợp theo nhóm:
     - mean, std, min, max, median, số fold
   - Tính thêm xếp hạng và độ ổn định:
     - `cv_stability = 1 - std_accuracy`
     - rank dense theo mean accuracy giảm dần
   - Lưu:
     - file plot PNG
     - báo cáo text gồm metadata + executive summary + bảng tổng hợp có thứ hạng + bảng chi tiết từng fold.

6. **So sánh fit time giữa 2 wrapper (`plot_fit_time_comparison`)**  
   - Đọc 2 file metrics CSV (seeded vs sklearn).
   - Kiểm tra cột bắt buộc `total_fit_time_ms`.
   - Tạo bảng so sánh (ms và đổi sang giây).
   - Vẽ biểu đồ 2 cột và annotate thời gian.
   - Lưu biểu đồ vào thư mục evaluation plots (hoặc `save_dir` nếu có truyền).

7. **Quy ước so sánh Raw vs Union**  
   Trong notebook so sánh wrapper, nhãn method nên chứa rõ biến thể dữ liệu và prefix report/plot nên gắn `raw` hoặc `union` để tránh ghi đè và giữ truy vết rõ ràng.

**4. Đầu ra / Artifacts:**  
- Artifacts về accuracy:
  - `.../plots/<experiment_prefix>_<dataset>.png`
  - `.../reports/<experiment_prefix>_<dataset>.txt`
  - Báo cáo gồm:
    - cấu hình method/model tốt nhất
    - thống kê tổng hợp theo method/model
    - kết quả từng fold để audit
- Artifacts về thời gian:
  - `.../plots/time_comparison_*.png`
  - bảng so sánh trả về gồm:
    - `algorithm`
    - `total_fit_time_ms`
    - `total_fit_time_sec`
- Phụ thuộc upstream cho phần time:
  - `metrics/metrics.csv` từ wrapper run-level:
    - `results/<dataset>/wrapper/raw/.../metrics/metrics.csv`
    - `results/<dataset>/wrapper/union/.../metrics/metrics.csv`
