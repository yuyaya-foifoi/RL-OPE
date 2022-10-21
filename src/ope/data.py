import numpy as np
import pandas as pd


def train_test_split(history: pd.DataFrame, method: str) -> tuple:

    if method == "half":
        train_dataset = history.iloc[: len(history) // 2]
        test_dataset = history.iloc[len(history) // 2 :]

        train_X = np.stack([x for x in train_dataset.State]).reshape(-1, 1)
        train_Y = np.array(train_dataset.Action).astype("int")
        test_X = np.stack([x for x in test_dataset.State]).reshape(-1, 1)
        test_Y = np.array(test_dataset.Action).astype("int")

    else:
        raise NotImplementedError()

    return (train_dataset, test_dataset, train_X, train_Y, test_X, test_Y)
