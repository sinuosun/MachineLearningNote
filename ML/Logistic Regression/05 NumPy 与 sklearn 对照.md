---
tags: [numpy, scikit-learn, implementation, classification]
---

# NumPy 与 sklearn 对照

| 概念 | 手写 NumPy | sklearn |
|---|---|---|
| 标准化 | `(X-mean)/std` | `StandardScaler()` |
| 多项式映射 | `polynomial_features` | `PolynomialFeatures` |
| 概率 | `sigmoid(X @ w + b)` | `predict_proba` |
| 类别 | `p >= threshold` | `predict` |
| 求解 | 批量梯度下降 | `lbfgs` 等优化器 |
| L2 强度 | `reg_lambda` 越大越强 | `C` 越小通常越强 |

```python
model = make_pipeline(
    StandardScaler(),
    PolynomialFeatures(degree=6, include_bias=False),
    LogisticRegression(C=20, max_iter=10_000, solver="lbfgs"),
)
model.fit(X_train, y_train)
```

## 关于 `C = 1/λ` 的精确程度

`C` 与正则化强度反向变化，写成 $C\propto1/\lambda$ 最安全。不同实现对数据项取“总和”还是“平均”、正则项系数以及截距处理可能不同，因此直接设 `C=1/reg_lambda` 是便于比较的起点，**不保证系数逐项相等**。更应比较测试指标、概率和决策边界。

## Pipeline 为什么重要

`Pipeline` 在每个训练折内部拟合标准化器和特征映射，能避免交叉验证中的数据泄漏。手写代码要显式做到同一件事：只用训练集统计量，然后复用于测试集和绘图网格。

完整实现：[[Code/logistic_regression.py]]。

VS Code 请直接打开本 Obsidian 仓库中的 [[Code/logistic_regression.py|配套代码]]。当前版本保留了你设置的六阶多项式思路，并补上稳定损失、固定随机种子、划分、标准化、早停、sklearn 基线和诊断图。

下一步：[[06 评估指标与模型诊断]]。
