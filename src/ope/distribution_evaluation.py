import numpy as np
import pandas as pd


def eval_policy_distance(
    test_dataset: pd.DataFrame,
    estimate_policy: np.array,
    predict_probs: np.array,
    brier_thresh: float,
    coef,
    distace_func: str,
):
    if distace_func == "total":
        return total_variation_distance_score(
            test_dataset, estimate_policy, coef
        )
    if distace_func == "brier":
        return brier_score(test_dataset, predict_probs, brier_thresh)


def total_variation_distance_score(
    test_dataset: pd.DataFrame, estimate_policy: np.array, coef
):
    return np.mean(
        np.abs(test_dataset.Behavior_Policy - estimate_policy) * coef
    )


def brier_score(
    test_dataset: pd.DataFrame, predict_probs: np.array, thresh: float = 1.0
) -> np.array:
    """
    \frac{1}{T} * ( \sum_{t}^{T} \sum_{i}^{I} (f_ti - o_ti) ^2)
    T : data 数
    I : action 数
    """
    o_matrix = _get_one_hot(test_dataset, predict_probs, thresh)
    return np.mean((predict_probs - o_matrix) ** 2)


def _get_one_hot(
    test_dataset: pd.DataFrame, predict_probs: np.array, thresh: float = 1.0
) -> np.array:
    """
    \frac{1}{T} * (\sum_{t}^{T} \sum_{i}^{I} (f_ti - o_ti) ^2)

    T : data 数
    I : action 数

    brier_scoreでは上記を計算する.
    しかし, off policy における履歴データはあるactionを取った時のそのactionに対するrewardしか得られない.
    -> 履歴データに残るaction以外の o_ti は分からない.
    -> 分からないので, rewardを考え成功とみなせるなら1, それ以外なら0とする
    """
    action_dim = len(test_dataset.Action.unique())
    o_matrix = np.zeros((len(test_dataset), action_dim))

    for data_idx in np.arange(len(test_dataset)):
        data = test_dataset.iloc[data_idx, :]
        # prob = predict_probs[data_idx, :]
        if data.Reward >= thresh:
            o_matrix[data_idx, :] = np.eye(action_dim)[int(data.Action)]
        else:
            o_matrix[data_idx, :] = (
                np.eye(action_dim)[int(data.Action)] * 0
            )  # 実質的にnp.zerosと同じ

    return o_matrix
