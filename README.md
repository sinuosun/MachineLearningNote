# MachineLearningNote

以 Obsidian 双链组织的 AI 与 CS 学习库，按“直觉 → 数学 → 实现 → 诊断 → 关联”逐步复现机器学习。

## 从这里开始

- [学习库首页](00%20Home.md)
- [机器学习概念地图](ML/00%20ML%20Index.md)
- [学习路线：从监督到无监督](ML/01%20Learning%20Roadmap.md)
- [线性回归主题](ML/Linear%20Regression/00%20Linear%20Regression%20MOC.md)
- [逻辑回归主题](ML/Logistic%20Regression/00%20Logistic%20Regression%20MOC.md)
- [神经网络入门](ML/Neural%20Network/00%20Neural%20Network%20MOC.md)

## 当前完成内容

- **线性回归**：多特征模型、MSE、$X^T$、梯度下降、正规方程、NumPy/sklearn 和三特征条件切片。
- **逻辑回归**：Sigmoid、log-odds、BCE 推导、多项式特征、L2、数据泄漏防护、阈值、混淆矩阵与概率诊断。
- **神经网络入门**：用户的二维二分类训练/测试数据、Keras 实验及泛化评估。完整反向传播推导待后续复现。

使用 Obsidian 打开仓库根目录即可浏览双链与关系图谱。配套 Python 文件可以独立运行并重新生成笔记中的图片。

虚拟环境、Python 缓存和 VS Code 临时工作区不纳入仓库。使用 VS Code 时只打开此仓库根目录，不再同时打开旧的副本。
