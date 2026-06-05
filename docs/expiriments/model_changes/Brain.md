# Brain Model Changes Expiriments

## Report

runing in raw variant

- Fully report is in: `Brain/evaluation/reports/benchmark_accuracy_raw_Brain.txt`

CROSS-VALIDATION SUMMARY (ranked)
| rank |Method |Model| mean_accuracy| std_accuracy| median_accuracy| min_accuracy| max_accuracy | n_folds| cv_stability|
|-|-|-|-|-|-|-|-|-|-|
|1| SFS_LOG| LogReg |0.9773| 0.0455| 1.0000| 0.9091| 1.0000| 4| 0.9545|
|2| SFS_RF| LogReg| 0.8818| 0.0426| 0.9000| 0.8182| 0.9091| 4| 0.9574|
|3| SFS_DT| Tree| 0.8568| 0.0557| 0.8591| 0.8000| 0.9091| 4| 0.9443|
|4| SFS_SVM| LogReg| 0.8364| 0.0823| 0.8591| 0.7273| 0.9000| 4| 0.9177|
|5| SFS_SVM| Tree| 0.8068| 0.1780| 0.8136| 0.6000| 1.0000| 4| 0.8220|
|5| None| LogReg| 0.8068| 0.0857| 0.8091| 0.7000| 0.9091| 4| 0.9143|
|6| SFS_DT| LogReg| 0.7591| 0.0682| 0.7591| 0.7000| 0.8182| 4| 0.9318|
|7| None| Tree| 0.7568| 0.1711| 0.7136| 0.6000| 1.0000| 4| 0.8289|
|8| SFS_RF| Tree| 0.7432| 0.1973| 0.8091| 0.4545| 0.9000| 4| 0.8027|
|9| SFS_LOG| Tree| 0.6841| 0.1765| 0.6636| 0.5000| 0.9091| 4| 0.8235|

- Time:

| Model | Selected_Features | Internal_SFS_Score | Time (s)           |
| ----- | ----------------- | ------------------ | ------------------ |
| LOG   | 6                 | 0.9772727272727272 | 118.78668149199802 |
| DT    | 6                 | 0.9045454545454544 | 50.683944659998815 |
| RF    | 6                 | 0.9772727272727272 | 2620.808770978001  |
| SVM   | 11                | 1.0                | 120.18944878199908 |

## Chart

![benchmark_accuracy_raw_Brain](../../results/Brain/evaluation/plots/benchmark_accuracy_raw_Brain.png)

![time](../../results/Brain/evaluation/plots/benchmark_time_features_raw_Brain.png)
