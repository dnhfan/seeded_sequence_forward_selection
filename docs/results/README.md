# Results and Evaluation

_Read this in [Vietnamese](README.vi.md)_

This index splits the previous monolithic report into one file per dataset.
You can find out how we evaluate in [5.Evaluation](../pipelines/05_evaluation_metrics.md)

## Executive Summary

Updated rankings from Final Evaluation (all methods comparison). See Section 10 in individual dataset reports for detailed method breakdowns.

| Dataset | Best Method + Model | Accuracy | Std | Features | Fit Time (ms) | 2nd Best + Model | 2nd Acc | 3rd Best + Model | 3rd Acc |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | --- | ---: |
| Lymphoma | ANOVA_F_TEST + LogReg | **1.0000** | 0.0000 | 2 | 3,496 | CHI_SQUARED + LogReg | 1.0000 | MUTUAL_INFORMATION + LogReg | 1.0000 |
| SRBCT_txt | ANOVA_F_TEST + LogReg | **1.0000** | 0.0000 | 4 | 4,076 | CHI_SQUARED + LogReg | 1.0000 | MUTUAL_INFORMATION + LogReg | 1.0000 |
| DLBCL | MUTUAL_INFORMATION + LogReg | 0.9875 | 0.0280 | 3 | 6,687 | Seeded_SFS_Raw + LogReg | 0.9867 | None + LogReg | 0.9733 |
| Lung_cancer | CHI_SQUARED + LogReg | 0.9784 | 0.0102 | 7 | 2,980 | CORRELATION + LogReg | 0.9770 | ANOVA_F_TEST + LogReg | 0.9741 |
| Leukemia_3c1 | ANOVA_F_TEST + LogReg | 0.9724 | 0.0379 | 2 | 2,849 | CHI_SQUARED + LogReg | 0.9724 | MUTUAL_INFORMATION + LogReg | 0.9724 |
| Leukemia_4c1 | Sklearn_SFS_Raw + Tree | 0.9722 | 0.0321 | 3 | 370,855 | Sklearn_SFS_Union + Tree | 0.9722 | ANOVA_F_TEST + LogReg | 0.9583 |
| Prostate | Seeded_SFS_Raw + LogReg | 0.9714 | 0.0602 | 4 | 63,907 | Seeded_SFS_Union + LogReg | 0.9614 | CHI_SQUARED + LogReg | 0.9514 |
| adenocarcinoma | Sklearn_SFS_Raw + LogReg | 0.9600 | 0.0365 | 3 | 644,416 | Sklearn_SFS_Union + LogReg | 0.9467 | Seeded_SFS_Union + LogReg | 0.9350 |
| Brain | ANOVA_F_TEST + LogReg | 0.9523 | 0.0552 | 6 | 3,554 | CORRELATION + LogReg | 0.9523 | Sklearn_SFS_Raw + Tree | 0.9295 |
| colon1 | Seeded_SFS_Raw + LogReg | 0.9179 | 0.0963 | 8 | 188,277 | Sklearn_SFS_Raw + LogReg | 0.9026 | CHI_SQUARED + LogReg | 0.8872 |
| Breast2classes | Seeded_SFS_Raw + LogReg | 0.9083 | 0.0381 | 11 | 102,179 | Sklearn_SFS_Raw + LogReg | 0.8833 | Sklearn_SFS_Union + LogReg | 0.8058 |
| CNS1 | ANOVA_F_TEST + LogReg | 0.9083 | 0.1387 | 5 | 22,329 | CORRELATION + LogReg | 0.9083 | Sklearn_SFS_Union + LogReg | 0.8833 |
| NCI | MUTUAL_INFORMATION + LogReg | 0.8692 | 0.0733 | 6 | 7,373 | ANOVA_F_TEST + LogReg | 0.8526 | CORRELATION + LogReg | 0.8359 |
| Breast3classes | Sklearn_SFS_Raw + LogReg | 0.7684 | 0.1026 | 5 | 370,924 | Seeded_SFS_Raw + LogReg | 0.7474 | Sklearn_SFS_Union + LogReg | 0.7368 |
| Tumors9 | Seeded_SFS_Union + LogReg | 0.7333 | 0.0385 | 11 | 4,536 | MUTUAL_INFORMATION + LogReg | 0.7333 | ANOVA_F_TEST + LogReg | 0.6667 |

## Dataset Reports

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

## Notes

- Brain report naming/reference mismatch remains in source artifacts.
