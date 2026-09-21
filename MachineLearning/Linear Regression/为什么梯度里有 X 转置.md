---
tags: [linear-algebra, gradient, intuition]
aliases: [为什么有 X^T, X 转置]
---

# 为什么梯度里有 $X^T$？

关键不是“为了套公式”，而是梯度要完成一次**反向汇总**。

## 前向：特征空间 → 样本空间

$$X_{n\times d}w_{d\times1}=\hat y_{n\times1}$$

$X$ 把 $d$ 个参数的影响送到 $n$ 个样本，得到每个样本的预测。

## 反向：样本误差 → 参数空间

残差 $e=\hat y-y$ 有 $n$ 个分量，但 $w$ 的梯度必须有 $d$ 个分量。因此需要一个 $d\times n$ 的映射：

$$X^T_{d\times n}e_{n\times1}=\nabla_wL_{d\times1}$$

第 $j$ 个输出正好是：

$$[X^Te]_j=\sum_{i=1}^n x_{ij}e_i$$

它回答：**第 $j$ 个特征在所有样本上，和误差共同变化了多少？**

- 若 $x_{ij}$ 大时误差也偏正，这个权重应向减小损失的方向调整。
- 若正负贡献抵消，当前损失对该权重不敏感。

```mermaid
flowchart LR
    W[参数空间 d] -- X --> E[样本预测/误差空间 n]
    E -- Xᵀ：按特征汇总误差 --> G[参数梯度空间 d]
```

## 链式法则视角

$\hat y=Xw+b\mathbf1$ 对 $w$ 的 Jacobian 是 $X$。标量损失沿计算图反传时使用 Jacobian 的转置：

$$\nabla_wL=\left(\frac{\partial \hat y}{\partial w}\right)^T\nabla_{\hat y}L
=X^T\frac{2}{n}(\hat y-y)$$

> [!tip] 记忆法
> 前向用 $X$ 把参数“展开到样本”；反向用 $X^T$ 把样本误差“收回到特征”。维度也会强迫结果从 $n$ 回到 $d$。

关联：[[多特征线性模型与维度]] · [[MSE 与梯度]] · [[梯度下降与正规方程]]

