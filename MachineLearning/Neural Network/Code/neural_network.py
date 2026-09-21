"""User-provided two-feature binary classification data, curated Keras baseline.

Run from anywhere: python /absolute/path/to/neural_network.py
Requires numpy and tensorflow. The original experiment is NeuralNetwork_original.
"""

from pathlib import Path

import numpy as np
import tensorflow as tf


DATA_DIR = Path(__file__).resolve().parent.parent / "Data"
SEED = 42
THRESHOLD = 0.5


def load_split(filename: str) -> tuple[np.ndarray, np.ndarray]:
    rows = np.loadtxt(DATA_DIR / filename, dtype=np.float32)
    if rows.ndim != 2 or rows.shape[1] != 3:
        raise ValueError(f"{filename}: expected rows of x1 x2 label")
    labels = rows[:, 2]
    if not np.isin(labels, [0, 1]).all():
        raise ValueError(f"{filename}: labels must be 0 or 1")
    return rows[:, :2], labels


def build_model() -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(2,)),
            tf.keras.layers.Dense(16, activation="tanh"),
            tf.keras.layers.Dense(8, activation="tanh"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy")],
    )
    return model


def main() -> None:
    tf.keras.utils.set_random_seed(SEED)
    x_train, y_train = load_split("trainDL_v1.txt")
    x_test, y_test = load_split("testDL_v1.txt")
    model = build_model()
    model.fit(x_train, y_train, epochs=200, batch_size=16, verbose=0)

    train_loss, train_acc = model.evaluate(x_train, y_train, verbose=0)
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    probabilities = model.predict(x_test, verbose=0).ravel()
    predictions = (probabilities >= THRESHOLD).astype(np.int32)
    labels = y_test.astype(np.int32)
    matrix = np.zeros((2, 2), dtype=np.int32)
    np.add.at(matrix, (labels, predictions), 1)

    print(f"train: n={len(y_train)}, BCE={train_loss:.4f}, accuracy={train_acc:.3f}")
    print(f"test:  n={len(y_test)}, BCE={test_loss:.4f}, accuracy={test_acc:.3f}")
    print("confusion matrix (rows: actual 0/1; cols: predicted 0/1):")
    print(matrix)
    print("test probabilities:", np.round(probabilities, 3))


if __name__ == "__main__":
    main()
