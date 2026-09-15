---
tags: [numpy, scikit-learn, implementation]
---

# NumPy 与 sklearn 对照

| 概念 | 手写 NumPy | sklearn `LinearRegression` |
|---|---|---|
| 模型 | `X @ w + b` | `model.predict(X)` |
| 拟合 | 梯度下降或 `np.linalg.lstsq` | `model.fit(X, y)` |
| 权重 | `w` | `model.coef_` |
| 截距 | `b` | `model.intercept_` |
| MSE | `np.mean((pred-y)**2)` | `mean_squared_error(y, pred)` |
| $R^2$ | 手算 | `model.score(X, y)` |

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)
print(model.coef_, model.intercept_)
print(mean_squared_error(y, y_pred))
```

`LinearRegression` 做的是普通最小二乘，而不是逐轮调用你写的梯度下降。相同目标函数、不同求解过程；在这组满秩小数据上，两者应收敛到接近的参数。

当前样例的精确拟合为：

$$w=\left[\frac73,\frac73,1\right],\qquad b\approx0$$

完整、可运行且带诊断图的实现：[[Code/linear_regression_numpy.py]]。

VS Code 请直接打开本 Obsidian 仓库。原始练习另存为 [[ML/Linear Regression/Code/Experiments/LinearRegression_original.py|最初版本]]，不会再依赖旧副本的绝对路径。

> [!note] 工程提示
> sklearn 负责可靠基线，NumPy 手写用于理解。真实项目还需训练／验证划分、特征预处理、数据泄漏检查和泛化评估；训练集 MSE 为零不代表泛化完美。

下一步：同一套对照思路用在分类上，见 [[ML/Logistic Regression/05 NumPy 与 sklearn 对照]]。
