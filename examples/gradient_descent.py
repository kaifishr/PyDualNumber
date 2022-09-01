"""Automatic differentiation and gradient descent with dual numbers.

Toy example performs and visualizes vanilla gradient descent with 
dual numbers.
"""
import numpy as np
import matplotlib.pyplot as plt
from dualnumber.engine import Dual


def plot_curve(axes, x_min=-3.0, x_max=3.0):
    """Plots function f"""
    style_dict = dict(color="black", alpha=1.0, linewidth=0.5)

    for ax in axes.flatten():
        x = np.linspace(x_min, x_max, num=1000)
        y = f(x)
        ax.plot(x, y, **style_dict)


def f(x):
    return x**2


def gradient_descent(x, n_steps, lr, keep_every=1):

    # Initializing dual number with dual part set to 1.
    x = Dual(real=x, dual=1.0)

    x_hist = list()
    x_hist.append(x.real)

    for i in range(n_steps):

        # Simultaneous feedforward and gradient computation
        # with dual numbers.
        y = f(x)

        # Gradient descent
        x = x - lr * y.dual

        if (i + 1) % keep_every == 0:
            x_hist.append(x.real)

    return np.array(x_hist)


def main():

    n_steps = 10
    x_init = 3.0
    keep_every_n = 1

    fig, axes = plt.subplots(nrows=2, ncols=2)

    plot_curve(axes)

    lrs = [0.01, 0.2, 0.4, 0.8]
    for lr, ax in zip(lrs, axes.flatten()):
        x_hist = gradient_descent(
            x=x_init, n_steps=n_steps, lr=lr, keep_every=keep_every_n
        )
        ax.plot(x_hist, f(x_hist), "r--", linewidth=0.5)
        ax.plot(x_hist, f(x_hist), "ro", markersize=2.0, label=f"lr = {lr}")
        ax.legend()
        ax.grid(alpha=0.4)

    plt.tight_layout()
    plt.show()
    # plt.savefig("gradient_descent_visualization.png", dpi=90)
    plt.close(fig)


if __name__ == "__main__":
    main()
