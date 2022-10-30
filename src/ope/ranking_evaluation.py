import numpy as np


def _DCG(pred_score) -> float:
    dcg_score = 0.0
    dcg_score += pred_score[0]
    for idx in np.arange(1, len(pred_score)):
        dcg_score += pred_score[idx] / np.log2(idx + 1)

    return dcg_score


def nDCG(ope_rank: np.array, ranks_in_metric: np.array) -> float:
    """
    ranks_in_metric : あるメトリックにおけるアルゴリズムの順位 (小さい方が良い)

    example

    scores_per_algos
    > array([3., 2., 4., 5., 1., 6.])

    ranks_in_metric
    > array([1., 5., 3., 0., 4., 2.])

    np.argsort(ranks_in_metric)
    > array([3, 0, 5, 2, 4, 1])

    scores_per_algos[np.argsort(ranks_in_metric)]
    > array([5., 3., 6., 4., 1., 2.])

    1) np.argsort(ranks_in_metric) -> ranks_in_metricの順位順にアルゴのindexを取得
    2) scores_per_algos[np.argsort(ranks_in_metric)] -> アルゴごとにアルゴの点数を取得

    """

    n_algo = len(ranks_in_metric)

    # アルゴごとに点数を割り当てる (n_algo=4, の中でrank=0なら, score=4)
    scores_per_algos = np.abs(ope_rank - n_algo)
    true_scores = np.abs(np.arange(n_algo) - n_algo)
    pred_scores = scores_per_algos[np.argsort(ranks_in_metric)]

    return _DCG(pred_scores) / _DCG(true_scores)
