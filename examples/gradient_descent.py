"""Automatic differentiation and gradient descent with dual numbers.

Toy example performs and visualizes gradient descent with dual numbers.
"""
import numpy as np
import matplotlib.pyplot as plt

from dualnumber import Dual


def plot_curve(axes, x_min=-3.0, x_max=3.0):
    """Plots loss landscape."""

    style_dict = dict(color="black", alpha=1.0, linewidth=0.5)

    for axis in axes.flatten():
        pred = np.linspace(x_min, x_max, num=1000)
        loss = pred**2
        axis.plot(pred, loss, **style_dict)


class Model:
    """Toy model with single scalar weight."""

    def __init__(self, weight_init: float = 1.0):
        self.weight = Dual(real=weight_init, dual=1.0)
        self.grad = 0.0

    def forward(self, data: float = 1.0):
        """Basic forward method.

        Data can be considered as constant one.
        """
        return data * self.weight

    def loss(self, data: float = 1.0):
        """Squared error loss function."""
        out = (0.0 - self.forward(data))**2
        if isinstance(out, Dual):
            self.grad = out.dual
            return out.real
        else:
            out

def gradient_descent(weight_init, n_steps, learning_rate, keep_every_n_steps=1):
    """Method performs gradient descent."""

    # Initializing dual number with dual part set to 1.
    model = Model(weight_init=weight_init)

    weight_hist = []
    weight_hist.append(model.weight.real)

    for i in range(n_steps):

        # Forward and loss computation.
        # Gradients are automatically computed.
        model.loss()

        # Gradient descent
        model.weight = model.weight - learning_rate * model.grad

        if (i + 1) % keep_every_n_steps == 0:
            weight_hist.append(model.weight.real)

    return np.array(weight_hist)


def main():
    """Main method for gradient descent with dual numbers."""

    n_steps = 10
    weight_init = 3.0
    keep_every_n_steps = 1

    fig, axes = plt.subplots(nrows=2, ncols=2)

    plot_curve(axes)

    learning_rates = [0.01, 0.2, 0.4, 0.8]
    for learning_rate, axis in zip(learning_rates, axes.flatten()):
        weight_hist = gradient_descent(
            weight_init=weight_init,
            n_steps=n_steps,
            learning_rate=learning_rate,
            keep_every_n_steps=keep_every_n_steps
        )
        label = f"lr = {learning_rate}"
        axis.plot(weight_hist, weight_hist**2, "ro", markersize=2.0, label=label)
        axis.plot(weight_hist, weight_hist**2, "r--", linewidth=0.5)
        axis.legend()
        axis.grid(alpha=0.4)

    plt.tight_layout()
    plt.show()
    # plt.savefig("gradient_descent_visualization.png", dpi=90)
    plt.close(fig)


if __name__ == "__main__":
    main()
