# MachineLearningNote

以 Obsidian 双链组织的 AI 与 CS 学习库，按“直觉 → 数学 → 实现 → 诊断 → 关联”逐步复现机器学习。

## 从这里开始

- [学习库首页](00%20Home.md)
- [机器学习概念地图](MachineLearning/ML%20Index.md)
- [学习路线：从监督到无监督](MachineLearning/Learning%20Roadmap.md)
- [线性回归主题](MachineLearning/Linear%20Regression/Linear%20Regression%20MOC.md)
- [逻辑回归主题](MachineLearning/Logistic%20Regression/Logistic%20Regression%20MOC.md)
- [神经网络入门](MachineLearning/Neural%20Network/Neural%20Network%20MOC.md)
- [CNN 学习目录](MachineLearning/CNN/CNN%20MOC.md)

## 学习进度与仓库内容

- **线性回归**：已学过，仓库已整理多特征模型、MSE、$X^T$、梯度下降、正规方程、NumPy/sklearn 和三特征条件切片。
- **逻辑回归**：已学过，仓库已整理 Sigmoid、log-odds、BCE 推导、多项式特征、L2、数据泄漏防护、阈值、混淆矩阵与概率诊断。
- **神经网络 / MLP**：已学过，仓库已有入门笔记、二维二分类数据和 Keras 实验。前向传播、反向传播待复习。
- **决策树与随机森林**：已学过，笔记尚未整理到仓库。
- **CNN**：已收录第 00 讲「神经网络基础」。**Transformer** 是未来学习方向。

使用 Obsidian 将仓库根目录作为 Vault 打开，从 `00 Home.md` 进入。若脚本、数据或 Notebook 未显示，在「文件与链接」中启用「显示所有文件类型」。线性回归和逻辑回归的配套脚本会重新生成各自 `Assets/` 下的图片。

## 本机 Python 与 Colab

在 macOS / VS Code 中打开本仓库根目录，使用 Anaconda 创建 Python 3.11 的 conda 环境：

```bash
conda create --name MachineLearning python=3.11
conda activate MachineLearning
python -m pip install -r requirements.txt
```

如果环境已经创建过，打开终端后运行 `conda activate MachineLearning`。在 VS Code 用 `Python: Select Interpreter` 选择该环境。运行本地 Notebook 时，也在内核菜单选择同一环境。

运行配套脚本时，先激活环境，再从仓库根目录执行所需的命令：

```bash
python "MachineLearning/Linear Regression/Code/linear_regression_numpy.py"
python "MachineLearning/Logistic Regression/Code/logistic_regression.py"
python "MachineLearning/Neural Network/Code/neural_network.py"
```

验证当前命令使用的解释器：

```bash
python -c "import sys; print(sys.executable)"
```

输出应指向 `MachineLearning` conda 环境里的 Python。神经网络目录中的 `NeuralNetwork_original.py` 保留了原始实验，上面的运行命令使用整理版。

可查看[神经网络 Notebook 源文件](MachineLearning/Neural%20Network/Code/neural_network_colab.ipynb)或[在 Google Colab 打开 main 版](https://colab.research.google.com/github/sinuosun/MachineLearningNote/blob/main/MachineLearning/Neural%20Network/Code/neural_network_colab.ipynb)。在 Colab 中按顺序运行三个代码单元格，由云端环境克隆仓库、补齐所需依赖，再运行 Keras 实验。

Git 会忽略 Python 缓存、本地虚拟环境和 Obsidian 工作区状态。
