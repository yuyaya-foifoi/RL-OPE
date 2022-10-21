import numpy as np
import pandas as pd


def calc_error(estimate_V: float, true_V: float, method="relative"):
    if method == "relative":
        return _relative_estimation_error(estimate_V, true_V)


def _relative_estimation_error(estimate_V: float, true_V: float) -> float:
    return np.abs((estimate_V - true_V) / true_V)


def _IPS(test_data: pd.DataFrame):
    rho = test_data.Estimate_Policy / test_data.Behavior_Policy
    estimate_V = np.mean(test_data.Reward * rho)
    return estimate_V


def estimate_V(test_data: pd.DataFrame, method="IPS"):
    if method == "IPS":
        return _IPS(test_data)


def _get_policy(model, test_X, test_Y, model_type="agent"):

    if model_type == "agent":
        policy = []
        for state in test_X:
            action, dist = model.get_action(test_X[0].item())
            policy.append(dist[action.item()])
        return policy

    elif model_type == "sklearn":
        dist = model.predict_proba(test_X)
        return [d[test_Y[idx]] for idx, d in enumerate(dist)]


def execute_ope(
    test_dataset: pd.DataFrame,
    estimate_policy_history: pd.DataFrame,
    model,
    sample_size: int = 1000,
    n_len: int = 10,
    model_type="agent",
    v_estimator: str = "IPS",
    error_function: str = "relative",
):

    test_dataset_cp = test_dataset.copy()
    test_X = np.stack([x for x in test_dataset.State]).reshape(-1, 1)
    test_Y = np.array(test_dataset.Action).astype("int")

    test_dataset_cp["Estimate_Policy"] = _get_policy(
        model, test_X, test_Y, model_type
    )
    true_V = np.mean(estimate_policy_history.Reward)

    ope_list = []

    for _ in np.arange(n_len):

        subset_test_data = test_dataset_cp.sample(sample_size)
        pred_V = estimate_V(subset_test_data, v_estimator)
        error = calc_error(pred_V, true_V, error_function)
        ope_list.append(error)

    return ope_list
