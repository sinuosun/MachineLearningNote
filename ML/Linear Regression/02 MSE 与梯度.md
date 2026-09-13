---
tags: [regression, loss-function, gradient]
---

# MSE 与梯度

令残差向量 $e=\hat y-y=Xw+b\mathbf1-y$。

## 均方误差 MSE

$$L(w,b)=\frac1n\sum_{i=1}^n(\hat y_i-y_i)^2
=\frac1n e^Te$$

- **平方**让正负误差不会抵消，并重罚大误差。
- **取平均**让损失尺度不随样本数线性增长。
- 它对离群点敏感；若噪声重尾，可考虑 MAE 或 Huber loss。
- 从概率角度，在独立同方差高斯噪声假设下，最小化 MSE 等价于最大似然估计。

## 逐参数求导

对第 $j$ 个权重：

$$\frac{\partial L}{\partial w_j}
=\frac2n\sum_{i=1}^n(\hat y_i-y_i)\frac{\partial\hat y_i}{\partial w_j}
=\frac2n\sum_{i=1}^n x_{ij}(\hat y_i-y_i)$$

把所有 $j$ 的偏导叠起来：

$$\nabla_wL=\frac2nX^T(\hat y-y)$$

对截距，由于 $\partial\hat y_i/\partial b=1$：

$$\frac{\partial L}{\partial b}=\frac2n\sum_{i=1}^n(\hat y_i-y_i)=2\operatorname{mean}(\hat y-y)$$

若定义误差为 $y-\hat y$，公式整体会多一个负号；两种写法完全等价，只需前后一致。

分类任务里 MSE 让位给交叉熵，梯度形式同样化简为 $p-y$：见 [[ML/Logistic Regression/02 二元交叉熵与梯度推导]]。

转置的直觉与维度推导：[[03 为什么梯度里有 X 转置]]。
