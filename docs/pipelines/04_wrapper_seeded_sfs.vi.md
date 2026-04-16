# Pipeline Wrapper - Custom Seeded Forward Selection (Seeded SFS)

_Read in [English](04_wrapper_seeded_sfs.md)._

**1. Tổng quan / Mục đích:**  
Pipeline này chạy thuật toán Seeded Forward Selection tùy biến, khởi tạo từ các seed features do ensemble voting đề xuất và mở rộng tập đặc trưng theo chiến lược greedy với điểm CV. Thiết kế này giúp tăng hiệu quả tìm kiếm và độ ổn định so với forward selection khởi tạo ngẫu nhiên/không seed.

Bạn có thể đọc chi tiết hơn về thuật toán này ở [SeededForwardSelection](../seeded_sfs_core_vi.md)

**2. Đầu vào:**

- DataFrame đầu vào: cột đầu là target, các cột sau là features.
- File seed từ ensemble voting:
  - `data/processed/<dataset>/03_ensemble/top<n_features>_features_voting.csv`
  - bắt buộc có cột: `Feature`, `Votes`
- Tham số runtime chính:
  - `n_seeds` (số seed lấy từ file voting)
  - `max_features`
  - `patience` (dừng sớm khi không cải thiện global)
  - `model`, `scoring`, `cv`, `random_state`
  - `n_jobs` cho đánh giá candidate song song
- Script chạy:
  - `notebook/<dataset>/07_sfs-raw.py`
  - `notebook/<dataset>/07_sfs-union.py`
- Class/file chính:
  - `src/wrapper/seeded.py` (`SeededSFSSelector`)
  - `src/wrapper/forward_selection.py` (`SeededForwardSelection`)
  - `src/wrapper/base.py` (điều phối + lưu artifacts)
  - `src/utils/utils.py` (`load_seed_from_csv`, `validate_features`)

**3. Logic thực thi từng bước:**

1. **Khởi tạo wrapper và xác định seed file**  
   `SeededSFSSelector` dựng đường dẫn seed CSV từ `ProjectPath.ensemble_dir / voting_csv_name`.

2. **Tách dữ liệu đầu vào và truyền cấu hình**  
   `run_sfs()` ở base wrapper tách `[y, X]`, tạo dict tham số, rồi truyền toàn bộ tham số đặc thù Seeded SFS (`n_seeds`, `patience`, ...) vào selector lõi.

3. **Khởi tạo nội bộ của SeededForwardSelection**  
   Trong `fit()`:
   - resolve model instance (`get_model` nếu đầu vào là alias string),
   - map tên feature -> index,
   - nạp top-`n_seeds` từ voting CSV (hoặc từ list),
   - kiểm tra tất cả seed có tồn tại trong cột dữ liệu hiện tại.

4. **Đóng băng CV splits để so sánh công bằng**  
   `_build_cv()` tạo `StratifiedKFold` (hoặc `KFold`), sau đó precompute/freeze splits vào `_cv_splits_`.  
   Nhờ vậy mọi candidate đều được chấm trên cùng bộ fold.

5. **Tính baseline score từ seed set**  
   Tập seed khởi tạo được đánh giá trước.  
   Từ đó thiết lập:
   - `current_score`
   - `global_best_score`
   - `global_best_indices`

6. **Mở rộng đặc trưng theo vòng lặp forward**  
   Ở mỗi iteration:
   - tạo candidates = các feature chưa chọn,
   - chấm từng candidate theo tập (`selected + candidate`) bằng CV mean,
   - chạy chấm candidate song song bằng `joblib.Parallel(n_jobs=self.n_jobs)`,
   - chọn candidate tốt nhất và thêm vào tập chọn,
   - ghi nhận improvement theo bước + thời gian vòng lặp,
   - cập nhật global best + patience counter.

7. **Điều kiện dừng**  
   Vòng lặp dừng khi:
   - không còn candidate,
   - hoặc đạt `max_features`,
   - hoặc hết `patience` (không tạo global peak mới sau N vòng).

8. **Cơ chế rollback theo global best**  
   Tập đặc trưng cuối lấy từ `global_best_indices`, không bắt buộc là tập ở iteration cuối.  
   Điều này giúp loại bỏ các feature thêm vào muộn nhưng làm giảm chất lượng.

9. **Sinh report và trả về SFSResult**  
   Selector tạo báo cáo text chi tiết có log từng iteration và timing, sau đó trả về:
   - dataset đã chọn đặc trưng
   - danh sách đặc trưng đã chọn
   - tổng fit time
   - global best score

10. **Lưu artifacts theo cùng chuẩn wrapper**  
    Base wrapper lưu CSV output và toàn bộ artifacts thí nghiệm theo cấu trúc run chuẩn.

**4. Đầu ra / Artifacts:**

- Dataset sau chọn đặc trưng:
  - `data/processed/<dataset>/04_wrapper/<variant>/seededsfsselector/<dataset>_seededsfsselector_*.csv`
- Run artifacts:
  - `results/<dataset>/wrapper/<variant>/seededsfsselector/run_YYYYMMDD_HHMMSS/`
    - `history/history.txt` (log đầy đủ theo iteration)
    - `features/selected_features.csv`
    - `metrics/metrics.json`
    - `metrics/metrics.csv` (có `total_fit_time_ms`)
- Dữ liệu quan trọng đặc thù:
  - quỹ đạo score theo iteration
  - thời gian từng iteration
  - tập global-best được giữ làm output cuối
