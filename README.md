# MachineLearningNote

以 Obsidian 双链组织的 AI 与 CS 学习库，按“直觉 → 数学 → 实现 → 诊断 → 关联”逐步复现机器学习。

## 从这里开始

- [学习库首页](00%20Home.md)
- [机器学习概念地图](MachineLearning/ML%20Index.md)
- [学习路线：从监督到无监督](MachineLearning/Learning%20Roadmap.md)
- [线性回归主题](MachineLearning/Linear%20Regression/Linear%20Regression%20MOC.md)
- [逻辑回归主题](MachineLearning/Logistic%20Regression/Logistic%20Regression%20MOC.md)
- [神经网络入门](MachineLearning/Neural%20Network/Neural%20Network%20MOC.md)

## 当前完成内容

- **线性回归**：多特征模型、MSE、$X^T$、梯度下降、正规方程、NumPy/sklearn 和三特征条件切片。
- **逻辑回归**：Sigmoid、log-odds、BCE 推导、多项式特征、L2、数据泄漏防护、阈值、混淆矩阵与概率诊断。
- **神经网络入门**：用户的二维二分类训练/测试数据、Keras 实验及泛化评估。完整反向传播推导待后续复现。

使用 Obsidian 打开仓库根目录即可浏览双链与关系图谱。配套 Python 文件可以独立运行并重新生成笔记中的图片。

## 本机 Python 与 Colab

在 macOS / VS Code 中打开本仓库根目录，使用 Anaconda 创建 Python 3.11 的 conda 环境：

```bash
conda create --name MachineLearning python=3.11
conda activate MachineLearning
python -m pip install -r requirements.txt
python "MachineLearning/Neural Network/Code/neural_network.py"
```

在 VS Code 中选择 `MachineLearning` conda 环境作为 Python 解释器，并确保安装依赖和运行脚本时都处于该环境。conda 环境由 Anaconda 管理，不需要在仓库中创建 `.venv`；依赖清单、源代码、数据和图片会提交。

如果环境已经创建过，后续只需运行 `conda activate MachineLearning`。VS Code 使用 `Python: Select Interpreter` 选择 `MachineLearning` conda 环境；终端、运行按钮和 Notebook 应使用同一个环境。

运行 ML 目录下的脚本时，先激活环境，再从仓库根目录执行：

```bash
conda activate MachineLearning
python "MachineLearning/Neural Network/Code/neural_network.py"
python "MachineLearning/Logistic Regression/Code/logistic_regression.py"
```

验证当前命令使用的解释器：

```bash
python -c "import sys; print(sys.executable)"
```

输出应指向 Anaconda 的 `envs/MachineLearning` 目录。Google Colab 不读取本机 conda 环境，可在 Colab 打开[神经网络 Colab 笔记本](MachineLearning/Neural%20Network/Code/neural_network_colab.ipynb)；它会从公开 GitHub 仓库拉取代码与数据，再在云端运行。

Python 缓存和 VS Code 临时工作区不纳入仓库。使用 VS Code 时只打开此仓库根目录，不再同时打开旧的副本。
