"""Three-feature linear regression from scratch, with honest 4D-friendly views.

Run:
    python linear_regression_numpy.py

Dependencies:
    numpy, matplotlib
Optional comparison:
    scikit-learn
"""

from pathlib import Path

import numpy as np


X = np.array(
    [
        [1, 2, 1],
        [2, 1, 2],
        [3, 3, 1],
        [4, 2, 3],
        [5, 4, 2],
        [6, 3, 4],
    ],
    dtype=float,
)
y = np.array([8, 9, 15, 17, 23, 25], dtype=float)


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean squared error."""
    return float(np.mean((y_pred - y_true) ** 2))


def fit_gradient_descent(
    features: np.ndarray,
    targets: np.ndarray,
    learning_rate: float = 0.01,
    max_epochs: int = 50_000,
    tolerance: float = 1e-12,
) -> tuple[np.ndarray, float, list[float]]:
    """Fit y = Xw + b with full-batch gradient descent."""
    n_samples, n_features = features.shape
    weights = np.zeros(n_features, dtype=float)
    bias = 0.0
    history: list[float] = []

    for _ in range(max_epochs):
        predictions = features @ weights + bias
        error = predictions - targets
        loss = float(np.mean(error**2))
        history.append(loss)

        if loss < tolerance:
            break

        # X.T maps n sample errors back to d parameter gradients.
        grad_w = (2.0 / n_samples) * features.T @ error
        grad_b = 2.0 * float(np.mean(error))

        new_weights = weights - learning_rate * grad_w
        new_bias = bias - learning_rate * grad_b
        if np.linalg.norm(new_weights - weights) + abs(new_bias - bias) < tolerance:
            weights, bias = new_weights, new_bias
            break
        weights, bias = new_weights, new_bias

    return weights, bias, history


def fit_least_squares(
    features: np.ndarray, targets: np.ndarray
) -> tuple[np.ndarray, float]:
    """Stable normal-equation equivalent; avoids an explicit matrix inverse."""
    design = np.column_stack([features, np.ones(len(features))])
    theta, *_ = np.linalg.lstsq(design, targets, rcond=None)
    return theta[:-1], float(theta[-1])


def create_visualizations(
    features: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray,
    bias: float,
    output_dir: Path,
) -> None:
    """Create conditional 3D slices and prediction/residual diagnostics."""
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = features @ weights + bias

    x1_grid, x2_grid = np.meshgrid(
        np.linspace(features[:, 0].min() - 0.5, features[:, 0].max() + 0.5, 30),
        np.linspace(features[:, 1].min() - 0.5, features[:, 1].max() + 0.5, 30),
    )
    slice_values = sorted(np.unique(features[:, 2]))
    fig = plt.figure(figsize=(12, 9), constrained_layout=True)
    for panel, fixed_x3 in enumerate(slice_values, start=1):
        axis = fig.add_subplot(2, 2, panel, projection="3d")
        surface = weights[0] * x1_grid + weights[1] * x2_grid + weights[2] * fixed_x3 + bias
        axis.plot_surface(x1_grid, x2_grid, surface, alpha=0.35, cmap="Blues", edgecolor="none")
        mask = features[:, 2] == fixed_x3
        axis.scatter(
            features[mask, 0], features[mask, 1], targets[mask],
            color="#e45756", s=55, depthshade=False, label="samples on slice",
        )
        axis.set(title=f"Conditional slice: x3 = {fixed_x3:g}", xlabel="x1", ylabel="x2", zlabel="y")
        axis.legend(loc="upper left", fontsize=8)
    fig.suptitle("Three-feature linear regression: fixed-x3 slices", fontsize=15)
    fig.savefig(output_dir / "linear_regression_slices.png", dpi=180)
    plt.close(fig)

    residuals = targets - predictions
    numerically_zero = np.max(np.abs(residuals)) < 1e-8
    residuals_to_plot = np.zeros_like(residuals) if numerically_zero else residuals
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    limits = [min(targets.min(), predictions.min()) - 1, max(targets.max(), predictions.max()) + 1]
    left.scatter(targets, predictions, color="#2a9d8f", s=65)
    left.plot(limits, limits, "--", color="#6b7280", label="ideal: prediction = truth")
    left.set(xlabel="True y", ylabel="Predicted y", title="Prediction vs truth", xlim=limits, ylim=limits)
    left.legend()
    right.axhline(0, linestyle="--", color="#6b7280")
    right.scatter(predictions, residuals_to_plot, color="#e76f51", s=65)
    right.set(
        xlabel="Predicted y",
        ylabel="Residual (true - predicted)",
        title="Residual diagnostics (all ≈ 0)" if numerically_zero else "Residual diagnostics",
    )
    if numerically_zero:
        right.set_ylim(-0.5, 0.5)
    fig.savefig(output_dir / "predictions_residuals.png", dpi=180)
    plt.close(fig)


def compare_with_sklearn(features: np.ndarray, targets: np.ndarray) -> None:
    """Optional baseline comparison when scikit-learn is installed."""
    try:
        from sklearn.linear_model import LinearRegression
    except ImportError:
        print("scikit-learn is not installed; skipping sklearn comparison.")
        return

    model = LinearRegression().fit(features, targets)
    print("sklearn  :", model.coef_, model.intercept_)


if __name__ == "__main__":
    w_gd, b_gd, losses = fit_gradient_descent(X, y)
    w_ls, b_ls = fit_least_squares(X, y)
    pred_gd = X @ w_gd + b_gd

    print(f"shape     : X={X.shape}, y={y.shape}, w={w_gd.shape}")
    print("gradient  :", w_gd, b_gd, "MSE=", mse(y, pred_gd))
    print("lstsq     :", w_ls, b_ls, "MSE=", mse(y, X @ w_ls + b_ls))
    print("epochs    :", len(losses))
    compare_with_sklearn(X, y)

    script_dir = Path(__file__).resolve().parent
    # Inside the Obsidian vault, Code/ and Assets/ are siblings. In the VS Code
    # project, keep generated files beside the script in LinearRegression_assets/.
    assets = (
        script_dir.parent / "Assets"
        if script_dir.name == "Code"
        else script_dir / "LinearRegression_assets"
    )
    create_visualizations(X, y, w_ls, b_ls, assets)
    print("figures   :", assets)
