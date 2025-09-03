# 项目A - CTR预估流水线 D2 实验结果

## 实验环境
* **数据量**: 100,000 条
* **哈希特征维度**: 16,384

## 性能对比

| 模型 (Model) | 特征集 (Feature Set) | AUC | LogLoss |
| :--- | :--- | :--- | :--- |
| Logistic Regression | Original | 0.646152 | 0.443348 |
| XGBoost | Original | 0.701039 | 0.424913 |
| Logistic Regression | Hashed | 0.657101 | 0.437521 |
| XGBoost | Hashed | **0.719450** | **0.417715** |

## 初步结论
1.  在相同特征集下，XGBoost 模型的性能全面优于逻辑回归。
2.  引入高基数类别特征并使用特征哈希处理后，两种模型的性能均有提升，证明该特征工程策略有效。