---
tags: [MOC, supervised-learning, classification]
aliases: [Logistic Regression, 逻辑回归]
---

# 🧭 逻辑回归

逻辑回归是[[ML/00 ML Index|监督学习]]中的**分类**算法：先用 $z=Xw+b$ 计算线性分数，再经 Sigmoid 得到类别 1 的概率。它与[[ML/Linear Regression/00 Linear Regression MOC|线性回归]]共享“模型 → 损失 → 梯度 → 优化 → 诊断”的主干。

```mermaid
flowchart TD
    DATA[带标签数据 X, y] --> SPLIT[训练集 / 测试集]
    SPLIT --> SCALE[只用训练集统计量标准化]
    SCALE --> MAP[可选：多项式特征映射 φ/X]
    MAP --> Z[logit: z = φ/X w + b]
    Z --> SIG[Sigmoid: p = P/y=1|x]
    SIG --> BCE[二元交叉熵]
    BCE --> GRAD[梯度: Xᵀ/p-y]
    GRAD --> L2[L2 正则化]
    L2 --> GD[梯度下降]
    GD --> EVAL[测试集指标与决策边界]
```

## 阅读顺序

1. [[01 从回归到分类：Sigmoid、概率与阈值]]
2. [[02 二元交叉熵与梯度推导]]
3. [[03 多项式特征与非线性决策边界]]
4. [[04 L2 正则化、容量与过拟合]]
5. [[05 NumPy 与 sklearn 对照]]
6. [[06 评估指标与模型诊断]]
7. [[Code/logistic_regression.py|可运行代码]]

## 一页速记

| 对象 | 公式 | 含义 |
|---|---|---|
| logit | $z=Xw+b$ | 尚未压缩的实数分数 |
| 概率 | $p=\sigma(z)$ | $P(y=1\mid x)$ |
| 决策 | $\hat y=\mathbf1[p\ge t]$ | 阈值 $t$ 默认 0.5，但可按代价调整 |
| BCE | $-y\log p-(1-y)\log(1-p)$ | 伯努利负对数似然 |
| 梯度信号 | $dz=p-y$ | **不是 loss**，是 $\partial L/\partial z$ |

带 L2 的训练目标：

$$J(w,b)=\frac1m\sum_{i=1}^m\left[\log(1+e^{z_i})-y_i z_i\right]
+\frac{\lambda}{2m}\lVert w\rVert_2^2$$

$$\nabla_wJ=\frac1mX^T(p-y)+\frac\lambda m w,qquad
\frac{\partial J}{\partial b}=\operatorname{mean}(p-y)$$

> [!important] 固定特征映射后是凸优化
> 普通逻辑回归以及固定多项式映射后的逻辑回归，对 $w,b$ 的 BCE 目标都是凸的；加入 L2 仍保持凸性。边界可以对原始输入非线性，但模型对参数仍是线性的。
