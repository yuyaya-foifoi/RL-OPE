import numpy as np
import pandas as pd


def eval_policy_distance(
    test_dataset: pd.DataFrame,
    estimate_policy: np.array,
    coef,
    distace_func: str,
):
    if distace_func == "total":
        return _total(test_dataset, estimate_policy, coef)


def _total(test_dataset: pd.DataFrame, estimate_policy: np.array, coef):
    return np.mean(
        np.abs(test_dataset.Behavior_Policy - estimate_policy) * coef
    )
