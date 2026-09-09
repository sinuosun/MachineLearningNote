---
tags: [numpy, scikit-learn, implementation, classification]
---

# NumPy 与 sklearn 对照

| 概念 | 手写 NumPy | sklearn `LogisticRegression` |
|---|---|---|
| 线性打分 | `z = X_poly @ w + b` | 内部处理 |
| 激活 | `sigmoid(z)` | `model.predict_proba(X)` |
| 损失 | 交叉熵 + L2 | `penalty='l2'` |
| 正则强度 | `reg_lambda` | `C = 1/λ`（注意是倒数） |
| 权重 | `w` | `model.coef_` |
| 截距 | `b` | `model.intercept_` |
| 类别预测 | `a >= 0.5` | `model.predict(X)` |

```python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

model = make_pipeline(
    PolynomialFeatures(degree=4, include_bias=False),  # 对应 map_feature
    LogisticRegression(C=1 / 0.05, max_iter=10000),    # C = 1/λ
)
model.fit(X, y.ravel())
```

> [!important] C 是 λ 的倒数
> sklearn 用 `C` 表示正则强度的**反**：`C` 越大正则越弱。手写代码里 `reg_lambda=0.05` 对应 `C=20`。两套记号方向相反，是 sklearn 新手最常见的踩坑点。

`LogisticRegression` 默认用 L2 正则，求解器 `lbfgs` 是拟牛顿法，比手写的批量梯度下降收敛快得多；两者优化的目标（BCE + L2）一致，参数应接近。手写版 `epochs=100000` 正是因为朴素全批量 GD 在这个规模上需要更多轮次。

完整、可运行且画出决策边界的实现：[[Code/logistic_regression.py]]。

同一份代码也保存在 VS Code 项目中：[打开 `/Users/snow/项目/ML/LogisticRegression.py`](file:///Users/snow/%E9%A1%B9%E7%9B%AE/ML/LogisticRegression.py)。文件头部保留了最初 6 样本直线边界的旧版本（整段注释），方便对照"从直线到曲线"的演化。

> [!note] 工程提示
> 与[[ML/Linear Regression/05 NumPy 与 sklearn 对照|线性回归的对照笔记]]同样的结论：sklearn 负责可靠基线，手写用于理解。真实项目还应做训练/验证划分、特征标准化（GD 对尺度敏感，lbfgs 也受益）、类别不平衡检查。

至此逻辑回归闭环走完。下一站是模型评估与[[04 L2 正则化与过拟合|偏差/方差]]的深入。
