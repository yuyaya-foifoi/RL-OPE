import numpy as np

# def KL_divergence(behavior, estimate):
#    # to avoid zero, reverse KL
#    return np.sum(estimate * np.log((estimate/behavior)+1e-10))


def KL_divergence(behavior, estimate):
    estimate += 1e-10
    return np.sum(behavior * np.log((behavior / estimate)))


def total_variation_distance(behavior, estimate):
    return 1 / len(behavior) * np.sum(np.abs(behavior - estimate))
