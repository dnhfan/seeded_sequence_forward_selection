# Hướng dẫn sử dụng Strategy Benchmark CLI

## Tổng quan

So sánh 3 evaluation strategies (CV, TTS, CustomCV) trên cùng 1 dataset.
Script chạy baseline + tất cả file SFS → tạo combined comparison chart + summary CSV.

## Yêu cầu

- Python 3.10+
- pandas, scikit-learn, seaborn, matplotlib, imbalanced-learn
- Data đã qua pipeline: `data/processed/{dataset}/04_wrapper/{variant}/`

## Cấu trúc thư mục đầu vào

```
data/processed/{dataset}/04_wrapper/{variant}/
  └── {algorithm}/
       └── {dataset}_{algorithm}_{model}_{scoring}_{max}max_{seeds}seeds_{pat}pat_{cv}cv_{variant}.csv
```

Ví dụ:
```
data/processed/colon1/04_wrapper/union/
  ├── seededsfsselector/
  │    └── colon1_seededsfsselector_log_accuracy_20max_1seeds_5pat_5cv_union.csv
  └── sklearnsfsselector/
       └── colon1_sklearnsfsselector_logisticregression_accuracy_automax_5cv_union.csv
```

## Cấu trúc thư mục đầu ra

```
results/{dataset}/evaluation/
  ├── reports/
  │    └── strategy_benchmark_summary_{variant}_{dataset}.csv
  └── plots/
       └── strategy_comparison_{variant}_{dataset}.png
```

## Tham số CLI

### Bắt buộc

| Arg | Mô tả |
|-----|-------|
| `dataset` | Tên dataset (VD: `colon1`, `Brain`) |

### Tuỳ chọn

| Arg | Short | Mặc định | Mô tả |
|-----|-------|----------|-------|
| `--variant` | `-v` | `union` | Data variant: `raw` hoặc `union` |
| `--n-features` | `-n` | `50` | Số features đầu vào |
| `--cv-splits` | `-k` | `5` | Số folds cho CV/CustomCV |
| `--n-iter` | `-i` | `100` | Số lần lặp cho TTS |
| `--test-size` | `-t` | `0.3` | Tỷ lệ dữ liệu test (0-1) |
| `--output-dir` | `-o` | `results/{dataset}/evaluation/` | Thư mục lưu output |
| `--skip-baseline` | - | `false` | Bỏ qua đánh giá baseline |
| `--algorithms` | - | `None` | Lọc SFS theo algorithm name |
| `--models` | - | `None` | Lọc SFS theo model type |

## Ví dụ cơ bản

```bash
# Chạy tất cả SFS + baseline
python experiments/Benchmarks/eval_strategy_benchmarks.py colon1

# Chỉ chạy variant union, 50 features
python experiments/Benchmarks/eval_strategy_benchmarks.py colon1 -v union -n 50

# Tùy chỉnh CV folds và TTS iterations
python experiments/Benchmarks/eval_strategy_benchmarks.py colon1 -k 10 -i 200

# Lưu output ra thư mục riêng
python experiments/Benchmarks/eval_strategy_benchmarks.py colon1 -o /tmp/benchmark_output
```

## Ví dụ lọc file SFS

```bash
# Chỉ chạy seeded SFS
python experiments/Benchmarks/eval_strategy_benchmarks.py colon1 --algorithms seededsfsselector

# Chỉ chạy sklearn SFS
python experiments/Benchmarks/eval_strategy_benchmarks.py colon1 --algorithms sklearnsfsselector

# Chỉ chạy logistic regression
python experiments/Benchmarks/eval_strategy_benchmarks.py colon1 --models log logisticregression

# Chỉ chạy decision tree
python experiments/Benchmarks/eval_strategy_benchmarks.py colon1 --models dt decisiontreeclassifier

# Seeded SFS + decision tree
python experiments/Benchmarks/eval_strategy_benchmarks.py colon1 --algorithms seededsfsselector --models dt

# Bỏ qua baseline, chạy sklearn SFS + random forest
python experiments/Benchmarks/eval_strategy_benchmarks.py colon1 --skip-baseline --algorithms sklearnsfsselector --models rf
```

## Logic lọc file SFS

| Filter | Behavior |
|--------|----------|
| Không có filter | Chạy **TẤT CẢ** file SFS tìm được |
| Chỉ `--algorithms` | Giữ file chứa bất kỳ algorithm name nào |
| Chỉ `--models` | Giữ file chứa bất kỳ model name nào |
| Cả hai | **AND logic** — file phải thỏa cả algorithm **VÀ** model |

Matching không phân biệt hoa/thường (case-insensitive).

## Kết quả đầu ra

### Summary CSV

```
results/{dataset}/evaluation/reports/strategy_benchmark_summary_{variant}_{dataset}.csv
```

Cột:
| Column | Mô tả |
|--------|-------|
| `Source` | Tên method (VD: `Baseline`, `sklearnsfsselector_logisticregression`) |
| `Model` | Classifier (`LogReg`, `Tree`) |
| `Strategy` | Evaluation strategy (`cv`, `tts`, `custom_cv`) |
| `mean_acc` | Accuracy trung bình |
| `std` | Độ lệch chuẩn |
| `min` | Accuracy thấp nhất |
| `max` | Accuracy cao nhất |
| `n_fold` | Số fold/iterations |

### Comparison Chart

```
results/{dataset}/evaluation/plots/strategy_comparison_{variant}_{dataset}.png
```

- **Faceted** theo Model (LogReg / Tree) — mỗi model 1 subplot
- **X-axis**: Feature Selection Methods (Baseline, các SFS methods)
- **Hue**: Strategy (CV / TTS / CustomCV)
- **Bar height**: Mean accuracy
- **Error bar**: Std deviation

## Giải thích 3 Strategies

| Strategy | Mô tả | Tham số |
|----------|-------|---------|
| `cv` | Stratified K-Fold Cross Validation (sklearn) | `--cv-splits` (mặc định 5) |
| `tts` | Repeated Stratified Train/Test Split | `--n-iter` (mặc định 100), `--test-size` (mặc định 0.3) |
| `custom_cv` | Manual Stratified K-Fold (tự implement) | `--cv-splits` (mặc định 5) |

## Workflow mẫu

```bash
# Bước 1: Chạy benchmark trên dataset colon1, variant union
python experiments/Benchmarks/eval_strategy_benchmarks.py colon1 -v union

# Bước 2: Xem kết quả
cat results/colon1/evaluation/reports/strategy_benchmark_summary_union_colon1.csv

# Bước 3: Xem chart
open results/colon1/evaluation/plots/strategy_comparison_union_colon1.png

# Bước 4: Nếu muốn chạy lại với filter cụ thể
python experiments/Benchmarks/eval_strategy_benchmarks.py colon1 -v union \
  --algorithms seededsfsselector \
  --models log dt
```

## Troubleshooting

| Vấn đề | Giải pháp |
|--------|-----------|
| `No {variant} SFS files found` | Kiểm tra thư mục `data/processed/{dataset}/04_wrapper/{variant}/` có file CSV chưa |
| `ModuleNotFoundError` | Chạy `pip install -r requirements.txt` + cài thêm sklearn, seaborn, xgboost |
| `--cv-splits must be >= 2` | Tăng giá trị `--cv-splits` lên tối thiểu 2 |
| Chart quá nhiều methods | Script tự chuyển sang horizontal bar nếu > 6 methods |
