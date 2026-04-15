# Tài liệu Seeded SFS Core (Tiếng Việt)

## Mục lục

- [1) Mục đích](#1-mục-đích)
- [2) Bản đồ kiến trúc](#2-bản-đồ-kiến-trúc)
  - [Core algorithm](#core-algorithm)
  - [Lớp orchestration wrapper](#lớp-orchestration-wrapper)
- [3) Hợp đồng đầu vào](#3-hợp-đồng-đầu-vào)
- [4) Luồng chạy end-to-end](#4-luồng-chạy-end-to-end)
- [5) Hành vi core algorithm (SeededForwardSelection)](#5-hành-vi-core-algorithm-seededforwardselection)
- [6) Bảng tham số chính](#6-bảng-tham-số-chính)
- [7) Hợp đồng đầu ra](#7-hợp-đồng-đầu-ra)
- [8) Lệnh chạy mẫu](#8-lệnh-chạy-mẫu)
- [9) Lỗi thường gặp và gotchas](#9-lỗi-thường-gặp-và-gotchas)
- [10) Ghi chú về reproducibility và performance](#10-ghi-chú-về-reproducibility-và-performance)
  Tài liệu này mô tả chi tiết phần core của thuật toán Seeded Sequential Forward Selection (Seeded SFS) trong repository này: luồng chạy, cấu hình, đầu ra và các lưu ý quan trọng.

## 1) Mục đích

Seeded SFS là phương pháp wrapper selection bắt đầu từ tập seed feature đã được xếp hạng trước (từ voting), sau đó mở rộng bằng forward selection với điểm đánh giá cross-validation.

So với sklearn SFS, bản custom này có thêm:

- khởi tạo từ seed rõ ràng,
- cơ chế theo dõi global-best và chốt kết quả theo đỉnh cao tốt nhất,
- history theo từng iteration + thông tin timing,
- tích hợp lưu artifact thông qua wrapper framework.

## 2) Bản đồ kiến trúc

### Core algorithm

- `src/wrapper/forward_selection.py`
  - `SeededForwardSelection`: selector tương thích sklearn (`fit`, `transform`, support mask, history/report).

### Lớp orchestration wrapper

- `src/wrapper/seeded.py`
  - `SeededSFSSelector`: nối giữa dữ liệu đầu vào, seed path, thực thi thuật toán và tạo `SFSResult`.
- `src/wrapper/base.py`
  - `BaseWrapperSelector`: luồng `run_sfs(...)` dùng chung và lưu artifact.

### Utility hỗ trợ

- `src/utils/utils.py`
  - `load_seed_from_csv(...)`, `validate_features(...)`, `create_union_features(...)`.
- `src/wrapper/models.py`
  - model factory (`logistic`, `rf`, `svm`, `xgboost`, `dt`).
- `src/utils/experiment_paths.py`
  - tạo thư mục run theo pattern `results/.../run_YYYYMMDD_HHMMSS...`.
- `src/config.py`
  - quy ước path trung tâm thông qua `ProjectPath`.

### Script chạy thường dùng

- Raw variant: `notebook/<dataset>/07_sfs-raw.py`
- Union variant: `notebook/<dataset>/07_sfs-union.py`

## 3) Đặc tả đầu vào

### DataFrame đầu vào

- Cột đầu tiên phải là target label `y`.
- Các cột còn lại là feature `X`.

`BaseWrapperSelector.run_sfs(...)` mặc định tách:

- `y_in = df.iloc[:, 0]`
- `X_in = df.iloc[:, 1:]`

### File seed voting CSV

- Thường nằm ở `data/processed/<dataset>/03_ensemble/`.
- Bắt buộc có cột:
  - `Feature`
  - `Votes`
- Seed được lấy từ đầu file: `df["Feature"].head(n_seeds)`.

### Đầu vào theo variant

- `raw`: đọc từ `ProjectPath.raw_path`.
- `union`: thường tạo bởi `create_union_features(...)` từ kết quả filter rồi đưa vào Seeded SFS.

## 4) Luồng chạy end-to-end

1. Notebook set config (`data_name`, `n_features`, `voting_csv_name`, tham số SFS).
2. Tạo `SeededSFSSelector(...)` với `dataset_variant` (`raw` hoặc `union`).
3. `run_sfs(...)` trong `src/wrapper/base.py`:
   - tách dataframe thành `X` và `y`,
   - tạo `sfs_params`,
   - gọi `_execute_core(...)`.
4. `_execute_core(...)` trong `src/wrapper/seeded.py`:
   - xác định seed CSV path (`self.path.ensemble_dir / voting_csv_name`),
   - khởi tạo `SeededForwardSelection(...)`,
   - chạy `fit(X, y)` rồi `transform(X)`,
   - tạo dataframe cuối (`y + selected features`),
   - trả về `SFSResult`.
5. `BaseWrapperSelector._save_sfs_output(...)` lưu artifact:
   - wrapper csv,
   - history txt,
   - selected features csv,
   - metrics json/csv.

## 5) Core Algorithm Behavior (`SeededForwardSelection`)

### Khởi tạo

- Resolve model:
  - nếu `model` là string -> dùng `get_model(...)`.
  - nếu là estimator instance -> dùng trực tiếp.
- Load seed từ:
  - list tên feature, hoặc
  - CSV path qua `load_seed_from_csv(...)`.
- Validate seed phải tồn tại trong cột của `X` (`validate_features`).

### Chiến lược CV

- Tạo CV splitter một lần qua `_build_cv()`:
  - `StratifiedKFold` nếu `cv_stratified=True`, ngược lại `KFold`.
  - tính ổn định với `random_state` nếu `cv_shuffle=True`.
- Freeze các split vào `_cv_splits_` và tái sử dụng cho mọi lần score.

### Baseline

- Tính điểm baseline chỉ trên tập seed ban đầu.
- Khởi tạo:
  - `current_score`,
  - `global_best_score`,
  - `global_best_indices`.

### Lặp forward selection

Mỗi iteration:

1. Lấy danh sách candidate chưa được chọn.
2. Đánh giá song song mỗi candidate:
   - score(`selected + candidate`) bằng `cross_val_score(...)`.
3. Chọn candidate có score cao nhất cho iteration đó.
4. Thêm candidate vào selected hiện tại.
5. Cập nhật:
   - improvement (`best_score - current_score`),
   - global best + patience counter,
   - history row kèm timing.

### Điều kiện dừng

- Không còn candidate.
- Chạm `max_features`.
- Hết `patience` (không cải thiện global-best trong N iteration).

### Finalize

- Kết quả cuối được chốt theo `global_best_indices` (tốt nhất toàn bộ quá trình, không nhất thiết là iteration cuối).
- Thuộc tính fit quan trọng:
  - `selected_features_`, `global_best_score_`,
  - `history_`, `total_iter_time_ms_`, `iteration_time_ms_`,
  - `total_fit_time_ms_`.

## 6) Bảng tham số chính

### Default của `SeededForwardSelection`

- `n_seeds=1`
- `model="logistic"`
- `scoring="accuracy"`
- `cv=5`
- `cv_shuffle=True`
- `cv_stratified=True`
- `max_features=100`
- `patience=5`
- `random_state=42`
- `verbose=2`
- `n_jobs=-1` (song song cấp candidate)
- `using_timer=True`
- `unit="ms"`

### Default của `BaseWrapperSelector.run_sfs(...)`

- `max_features=20`
- `cv=5`
- `verbose=2`
- `model="logistic"`
- `scoring="accuracy"`
- `direction="forward"`

Tham số bổ sung (ví dụ `n_seeds`, `patience`) được truyền tiếp vào `SeededForwardSelection`.

## 7) Đặc tả đầu ra

### Dữ liệu wrapper đã xử lý

- Vị trí: `data/processed/<dataset>/04_wrapper/<variant>/<algorithm>/...csv`
- Nội dung: dataframe target + các feature được chọn.

### Artifact tracking thực nghiệm

- Pattern gốc: `results/<dataset>/wrapper/<variant>/<algorithm>/run_YYYYMMDD_HHMMSS[_tag]/`
- Thư mục con:
  - `history/`
  - `features/`
  - `metrics/`
  - `artifacts/`

### File được lưu

- `history/history.txt`
- `features/selected_features.csv`
- `metrics/metrics.json`
- `metrics/metrics.csv`

`metrics/metrics.csv` có trường `total_fit_time_ms`, được dùng cho time comparison trong phần evaluation.

## 8) Lệnh chạy mẫu

Chạy từ repo root để tránh lỗi path/import:

```bash
python notebook/colon1/07_sfs-raw.py
python notebook/colon1/07_sfs-union.py
```

Đa số script union sẽ tạo lại union features qua `create_union_features(...)` trước khi chạy Seeded SFS.

## 9) Lỗi thường gặp và gotchas

- Seed CSV thiếu hoặc sai schema:
  - không tìm thấy file -> `FileNotFoundError`
  - thiếu cột `Feature`/`Votes` -> `ValueError`
- Seed không tồn tại trong dataset -> `ValueError` từ `validate_features(...)`.
- `cv < 2` -> không hợp lệ, `_build_cv()` báo lỗi.
- Check dừng (`max_features`, `patience`) được gọi sau mỗi iteration, nên luôn có ít nhất một iteration được thử.
- Chưa có bước deduplicate seed rõ ràng; seed trùng có thể dẫn đến index trùng.
- Có file cũ `src/wrapper/old_SFS.py`; implementation đang dùng là `src/wrapper/forward_selection.py`.

## 10) Ghi chú về reproducibility và performance

- Reproducibility:
  - CV splits được freeze một lần cho mỗi lần `fit`,
  - `random_state` chi phối model và CV shuffle (nếu áp dụng).
- Performance:
  - evaluate candidate dùng `joblib.Parallel(n_jobs=self.n_jobs)`,
  - mỗi `cross_val_score(...)` bên trong để `n_jobs=1` để hạn chế nested parallel oversubscription.
