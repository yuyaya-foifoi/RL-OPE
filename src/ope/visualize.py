import matplotlib.pyplot as plt
import numpy as np


def visualize_plot(
    x: np.array,
    y: np.array,
    yerr: np.array,
    model_name: list,
    is_errorbar: bool,
    labels: list,
):

    if is_errorbar:
        plt.errorbar(
            x,
            y,
            yerr=yerr,
            capsize=4,
            fmt="o",
            ecolor="red",
            color="black",
        )
        for i, label in enumerate(model_name):
            plt.annotate(model_name[i], (x[i], y[i]))
        plt.xlabel(labels[0])
        plt.ylabel(labels[1])
        plt.show()
        plt.close()

    if not is_errorbar:
        plt.scatter(
            x,
            y,
        )
        for i, label in enumerate(model_name):
            plt.annotate(model_name[i], (x[i], y[i]))
        plt.xlabel(labels[0])
        plt.ylabel(labels[1])
        plt.show()
        plt.close()
