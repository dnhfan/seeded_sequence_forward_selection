# Results and Evaluation

This index splits the previous monolithic report into one file per dataset.

## Executive Summary

| Dataset | Best Variant | Best Method/Model | Mean Accuracy | Std | Selected Features | Wrapper Global Best Score | Wrapper Fit Time (ms) |
|---|---:|---|---:|---:|---:|---:|---:|
| Brain | Raw | Seeded / LogReg | 0.8795 | 0.0532 | 11 | 0.9295 | 112,591 |
| Breast2classes | Raw | Seeded / LogReg | 0.9083 | 0.0381 | 11 | 0.9350 | 102,179 |
| Breast3classes | Raw | Sklearn / LogReg | 0.7684 | 0.1026 | 5 | 0.8000 | 370,924 |
| CNS1 | Union | Seeded / LogReg | 0.8833 | 0.1264 | 5 | 0.9167 | 22,330 |
| DLBCL | Raw | Seeded / LogReg | 0.9867 | 0.0298 | 3 | 0.9867 | 104,516 |
| Leukemia_3c1 | Raw | Seeded / Tree | 0.9600 | 0.0894 | 2 | 0.9600 | 33,883 |
| Leukemia_4c1 | Raw | Sklearn / Tree | 0.9722 | 0.0321 | 3 | 0.9722 | 370,856 |
| Lung_cancer | Raw | Sklearn / Tree | 0.9607 | 0.0326 | 6 | 0.9607 | 1,226,250 |
| Lymphoma | Union | Seeded / LogReg | **1.0000** | **0.0000** | 2 | **1.0000** | 3,496 |
| NCI | Raw | Seeded / LogReg | 0.7872 | 0.0937 | 7 | 0.8833 | 223,827 |
| Prostate | Raw | Seeded / LogReg | 0.9714 | 0.0639 | 4 | 0.9714 | 63,907 |
| SRBCT_txt | Raw | Seeded / LogReg | 0.9846 | 0.0344 | 4 | 1.0000 | 38,541 |
| Tumors9 | Union | Seeded / LogReg | 0.7333 | 0.0471 | 11 | 0.6500 | 4,536 |
| adenocarcinoma | Union | Sklearn / LogReg | 0.9467 | 0.0298 | 2 | 0.9474 | 13,152 |
| colon1 | Raw | Seeded / LogReg | 0.9179 | 0.1021 | 8 | 0.9346 | 188,278 |

## Dataset Reports

- [colon1](results/result-colon1.md)
- [Prostate](results/result-prostate.md)
- [Lung_cancer](results/result-lung_cancer.md)
- [Brain](results/result-brain.md)
- [Breast2classes](results/result-breast2classes.md)
- [Breast3classes](results/result-breast3classes.md)
- [CNS1](results/result-cns1.md)
- [DLBCL](results/result-dlbcl.md)
- [Leukemia_3c1](results/result-leukemia_3c1.md)
- [Leukemia_4c1](results/result-leukemia_4c1.md)
- [Lymphoma](results/result-lymphoma.md)
- [NCI](results/result-nci.md)
- [SRBCT_txt](results/result-srbct_txt.md)
- [Tumors9](results/result-tumors9.md)
- [adenocarcinoma](results/result-adenocarcinoma.md)

## Notes

- Missing wrapper metrics: `results/CNS1/wrapper/union/sklearnsfsselector/run_20260412_115642`
- Brain report naming/reference mismatch remains in source artifacts.
