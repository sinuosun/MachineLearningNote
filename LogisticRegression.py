"""从零实现二分类逻辑回归，并与 scikit-learn 基线比较。

本例使用两个月牙形类别，先标准化原始特征，再做多项式特征映射，
最后以带 L2 正则化的批量梯度下降训练逻辑回归。

依赖：numpy、matplotlib、scikit-learn
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


RANDOM_STATE = 42
DEGREE = 6
LEARNING_RATE = 0.1
REG_LAMBDA = 0.05
MAX_EPOCHS = 30_000


def sigmoid(z: np.ndarray) -> np.ndarray:
    """数值稳定的 Sigmoid。"""
    z = np.clip(z, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-z))


def polynomial_features(x: np.ndarray, degree: int) -> np.ndarray:
    """把二维输入映射为 1..degree 阶的全部多项式项，不加入常数列。"""
    x1, x2 = x[:, 0], x[:, 1]
    terms = []
    for total_degree in range(1, degree + 1):
        for x2_power in range(total_degree + 1):
            x1_power = total_degree - x2_power
            terms.append((x1**x1_power) * (x2**x2_power))
    return np.column_stack(terms)


def binary_cross_entropy_from_logits(
    logits: np.ndarray,
    y_true: np.ndarray,
    weights: np.ndarray,
    reg_lambda: float,
) -> float:
    """稳定计算 BCE + lambda/(2m)||w||²，避免直接 log(0)。"""
    m = len(y_true)
    data_loss = np.mean(np.logaddexp(0.0, logits) - y_true * logits)
    regularization = reg_lambda * np.sum(weights**2) / (2.0 * m)
    return float(data_loss + regularization)


def fit_logistic_regression(
    x_train: np.ndarray,
    y_train: np.ndarray,
    learning_rate: float = LEARNING_RATE,
    reg_lambda: float = REG_LAMBDA,
    max_epochs: int = MAX_EPOCHS,
    tolerance: float = 1e-9,
    patience: int = 200,
) -> tuple[np.ndarray, float, list[float]]:
    """使用全批量梯度下降训练带 L2 正则化的逻辑回归。"""
    m, n_features = x_train.shape
    weights = np.zeros(n_features)
    bias = 0.0
    history: list[float] = []
    best_loss = np.inf
    unchanged_epochs = 0

    for epoch in range(max_epochs):
        logits = x_train @ weights + bias
        probabilities = sigmoid(logits)
        error = probabilities - y_train

        # X.T 把 m 个样本的误差信号汇总成 n_features 个参数梯度。
        grad_w = (x_train.T @ error) / m + (reg_lambda / m) * weights
        grad_b = float(np.mean(error))  # 截距通常不正则化

        weights -= learning_rate * grad_w
        bias -= learning_rate * grad_b

        if epoch % 10 == 0:
            updated_logits = x_train @ weights + bias
            loss = binary_cross_entropy_from_logits(
                updated_logits, y_train, weights, reg_lambda
            )
            history.append(loss)
            if best_loss - loss > tolerance:
                best_loss = loss
                unchanged_epochs = 0
            else:
                unchanged_epochs += 1
                if unchanged_epochs >= patience:
                    break

    return weights, bias, history


def predict_proba(x: np.ndarray, weights: np.ndarray, bias: float) -> np.ndarray:
    return sigmoid(x @ weights + bias)


def predict(
    x: np.ndarray, weights: np.ndarray, bias: float, threshold: float = 0.5
) -> np.ndarray:
    return (predict_proba(x, weights, bias) >= threshold).astype(int)


def prepare_features(
    x_train_raw: np.ndarray, x_test_raw: np.ndarray, degree: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """只用训练集统计量标准化，避免数据泄漏。"""
    mean = x_train_raw.mean(axis=0)
    scale = x_train_raw.std(axis=0)
    scale = np.where(scale == 0, 1.0, scale)
    x_train_scaled = (x_train_raw - mean) / scale
    x_test_scaled = (x_test_raw - mean) / scale
    return (
        polynomial_features(x_train_scaled, degree),
        polynomial_features(x_test_scaled, degree),
        mean,
        scale,
    )


def create_visualizations(
    x_raw: np.ndarray,
    y: np.ndarray,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    weights: np.ndarray,
    bias: float,
    history: list[float],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    x_min, x_max = x_raw[:, 0].min() - 0.5, x_raw[:, 0].max() + 0.5
    y_min, y_max = x_raw[:, 1].min() - 0.5, x_raw[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 350), np.linspace(y_min, y_max, 350)
    )
    grid_raw = np.column_stack([xx.ravel(), yy.ravel()])
    grid_scaled = (grid_raw - mean) / scale
    grid_poly = polynomial_features(grid_scaled, DEGREE)
    grid_probability = predict_proba(grid_poly, weights, bias).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(9, 6.5), constrained_layout=True)
    field = ax.contourf(
        xx, yy, grid_probability, levels=np.linspace(0, 1, 21), cmap="RdBu_r", alpha=0.72
    )
    ax.contour(xx, yy, grid_probability, levels=[0.5], colors="#111827", linewidths=2.2)
    for label, color in [(0, "#2563eb"), (1, "#dc2626")]:
        train_mask = y[train_indices] == label
        test_mask = y[test_indices] == label
        ax.scatter(
            x_raw[train_indices][train_mask, 0],
            x_raw[train_indices][train_mask, 1],
            c=color,
            s=34,
            edgecolors="white",
            linewidths=0.5,
            label=f"Class {label} · train",
        )
        ax.scatter(
            x_raw[test_indices][test_mask, 0],
            x_raw[test_indices][test_mask, 1],
            c=color,
            s=48,
            marker="x",
            linewidths=1.4,
            label=f"Class {label} · test",
        )
    fig.colorbar(field, ax=ax, label="P(y = 1 | x)")
    ax.set(
        title=f"Polynomial logistic regression · degree={DEGREE}, λ={REG_LAMBDA}",
        xlabel="Feature 1",
        ylabel="Feature 2",
    )
    ax.legend(ncol=2, fontsize=9)
    fig.savefig(output_dir / "logistic_regression_decision_boundary.png", dpi=180)
    plt.close(fig)

    test_scaled = (x_raw[test_indices] - mean) / scale
    test_poly = polynomial_features(test_scaled, DEGREE)
    test_probability = predict_proba(test_poly, weights, bias)
    test_prediction = (test_probability >= 0.5).astype(int)
    matrix = confusion_matrix(y[test_indices], test_prediction)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3), constrained_layout=True)
    axes[0].plot(np.arange(len(history)) * 10, history, color="#2563eb")
    axes[0].set(xlabel="Epoch", ylabel="Regularized BCE", title="Training loss")

    image = axes[1].imshow(matrix, cmap="Blues")
    for row in range(2):
        for col in range(2):
            axes[1].text(col, row, matrix[row, col], ha="center", va="center", fontsize=14)
    axes[1].set(
        xticks=[0, 1], yticks=[0, 1], xlabel="Predicted", ylabel="True", title="Test confusion matrix"
    )
    fig.colorbar(image, ax=axes[1], fraction=0.046)

    axes[2].hist(test_probability[y[test_indices] == 0], bins=12, alpha=0.7, label="True class 0")
    axes[2].hist(test_probability[y[test_indices] == 1], bins=12, alpha=0.7, label="True class 1")
    axes[2].axvline(0.5, color="#111827", linestyle="--", label="threshold = 0.5")
    axes[2].set(xlabel="Predicted P(y=1)", ylabel="Count", title="Probability separation")
    axes[2].legend(fontsize=8)
    fig.savefig(output_dir / "logistic_regression_diagnostics.png", dpi=180)
    plt.close(fig)


def main() -> None:
    x_raw, y = make_moons(n_samples=400, noise=0.20, random_state=RANDOM_STATE)
    all_indices = np.arange(len(x_raw))
    train_indices, test_indices = train_test_split(
        all_indices, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )
    x_train, x_test, mean, scale = prepare_features(
        x_raw[train_indices], x_raw[test_indices], DEGREE
    )
    y_train, y_test = y[train_indices], y[test_indices]

    weights, bias, history = fit_logistic_regression(x_train, y_train)
    test_probability = predict_proba(x_test, weights, bias)
    test_prediction = predict(x_test, weights, bias)

    # sklearn 的 C 与正则强度反向变化；1/lambda 是直观对照，不保证因目标缩放约定而参数逐项相等。
    sklearn_model = make_pipeline(
        StandardScaler(),
        PolynomialFeatures(degree=DEGREE, include_bias=False),
        LogisticRegression(C=1.0 / REG_LAMBDA, max_iter=10_000, solver="lbfgs"),
    )
    sklearn_model.fit(x_raw[train_indices], y_train)
    sklearn_probability = sklearn_model.predict_proba(x_raw[test_indices])[:, 1]
    sklearn_prediction = sklearn_model.predict(x_raw[test_indices])

    print(f"mapped features : {x_train.shape[1]}")
    print(f"epochs used     : {(len(history) - 1) * 10}")
    print(
        f"NumPy test      : accuracy={accuracy_score(y_test, test_prediction):.3f}, "
        f"log_loss={log_loss(y_test, test_probability):.3f}"
    )
    print(
        f"sklearn test    : accuracy={accuracy_score(y_test, sklearn_prediction):.3f}, "
        f"log_loss={log_loss(y_test, sklearn_probability):.3f}"
    )

    script_dir = Path(__file__).resolve().parent
    output_dir = (
        script_dir.parent / "Assets"
        if script_dir.name == "Code"
        else script_dir / "LogisticRegression_assets"
    )
    create_visualizations(
        x_raw,
        y,
        train_indices,
        test_indices,
        mean,
        scale,
        weights,
        bias,
        history,
        output_dir,
    )
    print(f"figures         : {output_dir}")


if __name__ == "__main__":
    main()
