---
tags: [python, tooling, reproducibility, colab]
---

# 🛠️ Python、VS Code 与 Google Colab 环境

## 截图中的问题

你运行的是 `/Users/snow/.local/bin/python3.11`，这个解释器当时没有 NumPy。系统还有 Homebrew Python 3.14，它是另一个环境；即使给 3.14 安装包，也不会自动让 3.11 找到。`pip` 不在 shell 路径中时，使用 **同一个 Python** 执行 `python -m pip`：

```bash
python -m pip install numpy        # 在已激活的 .venv 中
.venv/bin/python -m pip install -r requirements.txt  # 不激活也可用
```

`python3 pip install numpy` 会把 `pip` 当成要执行的 Python 文件，不是安装命令。

## 本机环境（VS Code）

在仓库根目录创建 `.venv`，使用 Python 3.11（本机已有）：

```bash
/Users/snow/.local/bin/python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -c "import numpy, pandas, scipy, sklearn, matplotlib, tensorflow; print('ML imports OK')"
```

`requirements.txt` 是可重新安装的依赖清单；`.venv` 是本机生成的环境，体积较大且与操作系统有关，所以由 `.gitignore` 排除。VS Code 应选择仓库的 `.venv/bin/python` 解释器，不要在一个解释器运行、另一个解释器安装。

当前清单覆盖：NumPy（向量化）、SciPy（科学计算）、pandas（表格数据）、Matplotlib（静态图）、scikit-learn（传统机器学习基线）、TensorFlow/Keras（[[ML/Neural Network/00 Neural Network MOC|神经网络]]）和 ipykernel（VS Code 中执行 Notebook）。不预装整套大型深度学习生态；按学习主题再添加。

## Google Colab 是独立的云端环境

Colab 是托管 Notebook 服务，代码在 Google 的运行时执行，而不是使用本机 `.venv`。官方说明它预装很多包，运行时也会更新；需要特定版本时应在 Notebook 内安装。这里的 [[ML/Neural Network/Code/neural_network_colab.ipynb|Colab 入门笔记本]]从本仓库拉取脚本和数据，再在云端检查/安装缺失依赖并运行。无需在 Mac 上安装一个名为 `google-colab` 的包来使用网站。

在 Colab 打开 Notebook 后，先确认运行时已连接，依次运行单元格。若以后仓库设为私有，匿名 `git clone` 将不再可用；不要把 GitHub 凭据写进笔记本或仓库。

继续：[[ML/00 ML Index|机器学习地图]]、[[ML/01 Learning Roadmap|学习路线]]。
