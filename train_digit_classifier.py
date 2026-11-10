"""Teach a small neural network to read handwritten digits (0 to 9).

Every idea used here is explained with pictures in README.md.
"""

import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras

SEED = 42  # fixed so every run starts from the same random weights
EPOCHS = 20


def load_data() -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    # Pixels arrive as 0 to 255; scaling to 0 to 1 keeps gradient descent stable
    return (x_train / 255.0, y_train), (x_test / 255.0, y_test)


def show_digits(images: np.ndarray, titles: list[str], heading: str) -> None:
    plt.figure(figsize=(8, 3.5))
    for i, (image, title) in enumerate(zip(images, titles)):
        plt.subplot(2, 5, i + 1)
        plt.imshow(image, cmap="gray")
        plt.title(title, fontsize=9)
        plt.axis("off")
    plt.suptitle(heading)
    plt.tight_layout()


def build_model() -> keras.Sequential:
    keras.utils.set_random_seed(SEED)
    model = keras.Sequential(
        [
            keras.Input(shape=(28, 28)),
            keras.layers.Flatten(),                         # 28x28 grid -> row of 784 pixels
            keras.layers.Dense(300, activation="relu"),     # hidden layer 1
            keras.layers.Dense(100, activation="relu"),     # hidden layer 2
            keras.layers.Dense(10, activation="softmax"),   # one probability per digit
        ]
    )
    # Labels are plain integers 0-9, so the "sparse" version of cross-entropy fits
    model.compile(loss="sparse_categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])
    return model


def plot_training(history: keras.callbacks.History) -> None:
    epochs = range(1, len(history.history["accuracy"]) + 1)
    plt.figure()
    plt.plot(epochs, history.history["accuracy"], "o-", label="Training accuracy")
    plt.plot(epochs, history.history["val_accuracy"], "s--", label="Validation accuracy")
    plt.xlabel("Epoch (one full pass over the training images)")
    plt.ylabel("Accuracy")
    plt.title("The network gets better every pass")
    plt.legend()


def main() -> None:
    (x_train, y_train), (x_test, y_test) = load_data()
    print(f"{x_train.shape[0]} training images, {x_test.shape[0]} test images, each {x_train.shape[1]}x{x_train.shape[2]} pixels")
    show_digits(x_train[:10], [f"label {d}" for d in y_train[:10]], "What the network learns from")

    model = build_model()
    model.summary()

    # Hold back the last 5,000 training images to watch for overfitting while training
    history = model.fit(
        x_train[:-5000], y_train[:-5000],
        validation_data=(x_train[-5000:], y_train[-5000:]),
        epochs=EPOCHS,
    )
    plot_training(history)

    _, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test accuracy on 10,000 unseen digits: {test_accuracy:.1%}")

    probabilities = model.predict(x_test[:10], verbose=0)
    guesses = probabilities.argmax(axis=1)  # pick the digit with the highest probability
    show_digits(x_test[:10], [f"guess {g} ({p.max():.0%})" for g, p in zip(guesses, probabilities)], "Predictions on unseen digits")

    plt.show()


if __name__ == "__main__":
    main()
