"""Automatic differentiation and gradient descent with dual numbers.

Toy example performs and visualizes gradient descent with dual numbers.
"""
import numpy as np
import matplotlib.pyplot as plt
from dualnumber import Dual


def plot_curve(axes, x_min=-3.0, x_max=3.0):
    """Plots loss function of model."""
    style_dict = dict(color="black", alpha=1.0, linewidth=0.5)

    for axis in axes.flatten():
        data = np.linspace(x_min, x_max, num=1000)
        pred = model(data)
        axis.plot(data, pred, **style_dict)


def loss(data):
    """Squared error loss function."""
    return (0.0 - data)**2


def model(data):
    """Toy model consisting of forward method."""
    out = loss(data)
    return out


def gradient_descent(data, n_steps, learning_rate, keep_every_n_steps=1):
    """Method performs gradient descent."""

    # Initializing dual number with dual part set to 1.
    data = Dual(real=data, dual=1.0)

    x_hist = []
    x_hist.append(data.real)

    for i in range(n_steps):

        # Simultaneous feedforward and gradient computation
        # with dual numbers.
        pred = model(data)

        # Gradient descent
        data = data - learning_rate * pred.dual

        if (i + 1) % keep_every_n_steps == 0:
            x_hist.append(data.real)

    return np.array(x_hist)


def main():
    """Main method for gradient descent with dual numbers."""

    n_steps = 10
    x_init = 3.0
    keep_every_n_steps = 1

    fig, axes = plt.subplots(nrows=2, ncols=2)

    plot_curve(axes)

    learning_rates = [0.01, 0.2, 0.4, 0.8]
    for learning_rate, axis in zip(learning_rates, axes.flatten()):
        x_hist = gradient_descent(
            data=x_init,
            n_steps=n_steps,
            learning_rate=learning_rate,
            keep_every_n_steps=keep_every_n_steps
        )
        label = f"lr = {learning_rate}"
        axis.plot(x_hist, model(x_hist), "ro", markersize=2.0, label=label)
        axis.plot(x_hist, model(x_hist), "r--", linewidth=0.5)
        axis.legend()
        axis.grid(alpha=0.4)

    plt.tight_layout()
    plt.show()
    # plt.savefig("gradient_descent_visualization.png", dpi=90)
    plt.close(fig)


if __name__ == "__main__":
    main()
