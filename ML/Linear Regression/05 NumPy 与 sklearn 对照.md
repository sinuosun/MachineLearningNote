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

同一份整理后代码也保存在 VS Code 项目中：[打开 `/Users/snow/项目/ML/LinearRegression.py`](file:///Users/snow/%E9%A1%B9%E7%9B%AE/ML/LinearRegression.py)。原先无扩展名的 `LinearRegression` 文件保留不动，方便回看最初版本。

> [!note] 工程提示
> sklearn 负责可靠基线，NumPy 手写用于理解。真实项目还需训练／验证划分、特征预处理、数据泄漏检查和泛化评估；训练集 MSE 为零不代表泛化完美。
