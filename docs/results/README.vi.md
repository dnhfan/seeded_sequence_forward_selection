# Kết Quả và Đánh Giá

_Read this in [English](README.md)_

Tài liệu tổng quan này tách báo cáo tổng hợp trước đây thành một file riêng cho từng dataset.
Bạn có thể xem cách đánh giá tại [5.Evaluation](../pipelines/05_evaluation_metrics.md)

## Tóm Tắt Điều Hành

| Dataset        | Biến thể tốt nhất | Phương pháp/Mô hình tốt nhất | Accuracy trung bình |        Std | Số đặc trưng chọn | Wrapper global best score | Thời gian fit wrapper (ms) |
| -------------- | ----------------: | ---------------------------- | ------------------: | ---------: | ----------------: | ------------------------: | -------------------------: |
| Brain          |               Raw | Seeded / LogReg              |              0.8795 |     0.0532 |                11 |                    0.9295 |                    112,591 |
| Breast2classes |               Raw | Seeded / LogReg              |              0.9083 |     0.0381 |                11 |                    0.9350 |                    102,179 |
| Breast3classes |               Raw | Sklearn / LogReg             |              0.7684 |     0.1026 |                 5 |                    0.8000 |                    370,924 |
| CNS1           |             Union | Seeded / LogReg              |              0.8833 |     0.1264 |                 5 |                    0.9167 |                     22,330 |
| DLBCL          |               Raw | Seeded / LogReg              |              0.9867 |     0.0298 |                 3 |                    0.9867 |                    104,516 |
| Leukemia_3c1   |               Raw | Seeded / Tree                |              0.9600 |     0.0894 |                 2 |                    0.9600 |                     33,883 |
| Leukemia_4c1   |               Raw | Sklearn / Tree               |              0.9722 |     0.0321 |                 3 |                    0.9722 |                    370,856 |
| Lung_cancer    |               Raw | Sklearn / Tree               |              0.9607 |     0.0326 |                 6 |                    0.9607 |                  1,226,250 |
| Lymphoma       |             Union | Seeded / LogReg              |          **1.0000** | **0.0000** |                 2 |                **1.0000** |                      3,496 |
| NCI            |               Raw | Seeded / LogReg              |              0.7872 |     0.0937 |                 7 |                    0.8833 |                    223,827 |
| Prostate       |               Raw | Seeded / LogReg              |              0.9714 |     0.0639 |                 4 |                    0.9714 |                     63,907 |
| SRBCT_txt      |               Raw | Seeded / LogReg              |              0.9846 |     0.0344 |                 4 |                    1.0000 |                     38,541 |
| Tumors9        |             Union | Seeded / LogReg              |              0.7333 |     0.0471 |                11 |                    0.6500 |                      4,536 |
| adenocarcinoma |             Union | Sklearn / LogReg             |              0.9467 |     0.0298 |                 2 |                    0.9474 |                     13,152 |
| colon1         |               Raw | Seeded / LogReg              |              0.9179 |     0.1021 |                 8 |                    0.9346 |                    188,278 |

## Báo Cáo Theo Dataset

- [colon1](./result-colon1.md)
- [Prostate](./result-prostate.md)
- [Lung_cancer](./result-lung_cancer.md)
- [Brain](./result-brain.md)
- [Breast2classes](./result-breast2classes.md)
- [Breast3classes](./result-breast3classes.md)
- [CNS1](./result-cns1.md)
- [DLBCL](./result-dlbcl.md)
- [Leukemia_3c1](./result-leukemia_3c1.md)
- [Leukemia_4c1](./result-leukemia_4c1.md)
- [Lymphoma](./result-lymphoma.md)
- [NCI](./result-nci.md)
- [SRBCT_txt](./result-srbct_txt.md)
- [Tumors9](./result-tumors9.md)
- [adenocarcinoma](./result-adenocarcinoma.md)

## Ghi Chú

- Thiếu wrapper metrics: `results/CNS1/wrapper/union/sklearnsfsselector/run_20260412_115642`
- Vẫn còn lệch tên/tham chiếu báo cáo Brain trong source artifacts.
