# Results and Evaluation

_Read this in [Vietnamese](README.vi.md)_

This index splits the previous monolithic report into one file per dataset.
You can find out how we evaluate in [5.Evaluation](../pipelines/05_evaluation_metrics.md)

SFS Results for this report is using logistic reg.
If you wanna see a benchmark where i ran SFS for all model pls go [SFS Benchmark](../expiriments/model_changes/README.md)

## Executive Summary

Updated rankings from Final Evaluation (all methods comparison). See Section 10 in individual dataset reports for detailed method breakdowns.

| DataSet        |   Top 1 Methods/variant   | Seeded Selected | Seeded Global Best | Seeded Fit Time (s) |
| :------------- | :-----------------------: | --------------: | -----------------: | ------------------: |
| Brain          | Seeded_SFS_Union + LogReg |               6 |           0.977273 |            5.791480 |
| colon1         |  Seeded_SFS_Raw + LogReg  |              10 |             0.9679 |  30.885568714002147 |
| adenocarcinoma |  Seeded_SFS_Raw + LogReg  |               4 |             1.0000 |   190.4245332389837 |
| Breast2classes |  Seeded_SFS_Raw + LogReg  |               9 |            0.93583 |  139.02734391699778 |
| Breast3classes |  Seeded_SFS_Raw + LogReg  |              17 |             0.8842 |            322.931 |
| CNS1           |  Seeded_SFS_Raw + LogReg  |               7 |             0.9833 |  175.75089231900347 |
| Leukemia_3c1   | Seeded_SFS_Union + LogReg |               4 |            1.000000 |              5.384 |
| Leukemia_4c1   | Seeded_SFS_Union + LogReg |               8 |            1.000000 |              6.948 |

## Dataset Reports

- [Brain](./result-brain.md)
- [colon1](./result-colon1.md)
- [Prostate](./result-prostate.md)
- [Lung_cancer](./result-lung_cancer.md)
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
