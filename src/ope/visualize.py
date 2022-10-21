import matplotlib.pyplot as plt
import numpy as np


def visualize_error_per_distance(opes: np.array, model_name: list):

    x = opes[:, 0]
    y = np.abs(np.mean(opes[:, 1:], axis=1))

    plt.errorbar(
        x,
        y,
        yerr=np.std(opes[:, 1:], axis=1),
        capsize=4,
        fmt="o",
        ecolor="red",
        color="black",
    )
    for i, label in enumerate(model_name):
        plt.annotate(model_name[i], (x[i], y[i]))
    plt.xlabel("distribution distance")
    plt.ylabel("value error")
    plt.show()
    plt.close()
