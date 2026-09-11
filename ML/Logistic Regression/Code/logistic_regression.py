import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

# 1. 生成规模更大且具备非线性边界的合成数据集 (200 个样本)
X, y = make_moons(n_samples=200, noise=0.20)
# 这里可以加上 random_state=42 来保证每次生成的数据集相同
y = y.reshape(-1, 1)

# 2. 特征映射函数: 多项式基函数展开 (Polynomial Feature Mapping)
# 将 [x1, x2] 映射为 [x1, x2, x1^2, x1*x2, x2^2, x1^3, ...]
def map_feature(x1, x2, degree=4):  # 这个也可以调整！
    out = []
    for i in range(1, degree + 1):  # i 表示阶数
        for j in range(i + 1):      # j 表示 x2 的指数
            out.append((x1 ** (i - j)) * (x2 ** j))
    return np.column_stack(out)

# 构造高阶特征矩阵
# ========================调整阶数========================
degree = 4  # 可调节阶数: 2 (二次), 3 (三次), 4 等
X_poly = map_feature(X[:, 0], X[:, 1], degree=degree)

# 3. 初始化参数与超参数
w = np.zeros((X_poly.shape[1], 1))
b = 0.0

# =========================调参=========================
alpha = 0.1         # 学习率
reg_lambda = 0.05   # L2 正则化系数 (lambda > 0 防止高阶项过拟合)
epochs = 100000


m = len(X)

def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -250, 250)))

# 4. 批量梯度下降 (含 L2 正则化梯度)
for _ in range(epochs):
    z = X_poly @ w + b
    y_pred = sigmoid(z)
    dz = y_pred - y  # 交叉熵 + Sigmoid 联合求导后的形式

    # 偏导数: dw 加入 L2 惩罚项 (注意: bias 项 b 通常不参与正则化)
    # 岭回归 Ridge Regression
    dw = (X_poly.T @ dz) / m + (reg_lambda / m) * w
    db = np.mean(dz)

    w -= alpha * dw
    b -= alpha * db

# 5. 绘制决策边界
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                     np.linspace(y_min, y_max, 300))

# 对网格点同样进行多项式映射
grid_poly = map_feature(xx.ravel(), yy.ravel(), degree=degree)
grid_z = grid_poly @ w + b
grid_predictions = sigmoid(grid_z).reshape(xx.shape)

plt.figure(figsize=(8, 6))

# 填充决策区域
plt.contourf(xx, yy, grid_predictions, levels=[0, 0.5, 1], alpha=0.25, colors=['blue', 'red'])
# 绘制 P(y=1|x) = 0.5 的非线性决策边界
plt.contour(xx, yy, grid_predictions, levels=[0.5], colors='black', linewidths=2)

# 绘制两类样本散点
plt.scatter(X[y[:, 0] == 0][:, 0], X[y[:, 0] == 0][:, 1], color='blue', edgecolor='k', label='Class 0', alpha=0.8)
plt.scatter(X[y[:, 0] == 1][:, 0], X[y[:, 0] == 1][:, 1], color='red', edgecolor='k', label='Class 1', alpha=0.8)

plt.title(f"Polynomial Logistic Regression (Degree = {degree}, $\\lambda$ = {reg_lambda})")
plt.xlabel("Feature 1 ($x_1$)")
plt.ylabel("Feature 2 ($x_2$)")
plt.legend()
plt.tight_layout()
plt.show()
