"""Regenerate the concept figures in README.md. These need no dataset download."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).parent / "images"
OUT.mkdir(parents=True, exist_ok=True)


def relu_figure() -> None:
    z = np.linspace(-4, 4, 200)
    plt.figure(figsize=(5, 3.2))
    plt.plot(z, np.maximum(0, z), lw=2.5)
    plt.axhline(0, color="grey", lw=0.6)
    plt.axvline(0, color="grey", lw=0.6)
    plt.title("ReLU: keep positives, zero out negatives")
    plt.xlabel("neuron's weighted sum")
    plt.ylabel("output")
    plt.tight_layout()
    plt.savefig(OUT / "relu.png", dpi=150)
    plt.close()


def softmax_figure() -> None:
    scores = np.array([0.4, -1.2, 1.1, 0.2, -0.5, 0.0, -1.0, 4.2, 0.3, 1.0])
    probs = np.exp(scores) / np.exp(scores).sum()
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    axes[0].bar(range(10), scores, color="#999999")
    axes[0].set_title("Raw scores from the last layer")
    axes[1].bar(range(10), probs, color=["#d1495b" if i == 7 else "#999999" for i in range(10)])
    axes[1].set_title(f"After softmax: sums to 1, digit 7 wins ({probs[7]:.0%})")
    for ax in axes:
        ax.set_xticks(range(10))
        ax.set_xlabel("digit")
    plt.tight_layout()
    plt.savefig(OUT / "softmax.png", dpi=150)
    plt.close()


def parameters_figure() -> None:
    layers = ["Hidden 1\n784 -> 300", "Hidden 2\n300 -> 100", "Output\n100 -> 10"]
    weights = np.array([784 * 300, 300 * 100, 100 * 10])
    biases = np.array([300, 100, 10])
    plt.figure(figsize=(6, 3.4))
    plt.bar(layers, weights, label="weights", color="#30638e")
    plt.bar(layers, biases, bottom=weights, label="biases", color="#edae49")
    for i, total in enumerate(weights + biases):
        plt.text(i, total, f"{total:,}", ha="center", va="bottom")
    plt.ylim(0, (weights + biases).max() * 1.15)
    plt.ylabel("learnable numbers")
    plt.title(f"{(weights + biases).sum():,} parameters, most in the first layer")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "parameters.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    relu_figure()
    softmax_figure()
    parameters_figure()
    print(f"Figures written to {OUT}")
