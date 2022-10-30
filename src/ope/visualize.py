import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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


def get_estimate_freq_and_actual_freq(x, action_idx):
    return np.median(x.prob), np.sum(x.Action == action_idx) / len(x)


def vis_multiclass_calibration_curve(
    test_dataset: pd.DataFrame,
    models: dict,
    data: tuple,
    n_bin: int = 20,
    fig_size: tuple = (18.0, 4.0),
):

    """
    あるaction クラスをpositive, それ以外をnegativeとしてcalibration curveを作成する.
    1) actionについてloopを回す
    2) 各actionについて,全てのmodelを学習し, 該当クラスのprobを得る
    3) 可視化

    modelsの例
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.calibration import CalibratedClassifierCV
    models = {
        'rf' : RandomForestClassifier(),
        'knn' : KNeighborsClassifier(),
        'lr' : LogisticRegression(),
        'rf_iso' : CalibratedClassifierCV(
            RandomForestClassifier(),
            cv=2,
            method="isotonic"
            ),
        'knn_iso' : CalibratedClassifierCV(
            KNeighborsClassifier(),
            cv=2,
            method="isotonic"
            ),
        'lr_iso' : CalibratedClassifierCV(
            LogisticRegression(),
            cv=2,
            method="isotonic"
            )
    }
    """
    train_X, train_Y, test_X = data
    test_dataset_cp = test_dataset.copy()
    bins = np.arange(10) / n_bin
    plt.figure(figsize=fig_size)

    for action_idx in np.unique(test_dataset_cp.Action):
        action_idx = int(action_idx)
        plt.subplot(1, len(np.unique(test_dataset_cp.Action)), action_idx + 1)
        plt.title("action idx {}".format(action_idx))
        all_x = []
        for model_idx in np.arange(len(models)):
            model = list(models.values())[model_idx]
            model_name = list(models.keys())[model_idx]

            model.fit(train_X, train_Y)
            predict_probs = model.predict_proba(test_X)
            test_dataset_cp["prob"] = predict_probs[:, action_idx]
            test_dataset_cp["bins"] = pd.cut(
                test_dataset_cp.prob, bins=bins, labels=False
            )
            result = test_dataset_cp.groupby("bins").apply(
                get_estimate_freq_and_actual_freq, action_idx=action_idx
            )

            xs, ys = [], []
            for r in result:
                x, y = r
                xs.append(x)
                ys.append(y)
            all_x.extend(xs)

            plt.plot(xs, ys, label=model_name)
        plt.plot(
            [0, np.max(all_x)],
            [0, np.max(all_x)],
            color="black",
            linestyle="dashed",
        )
        plt.legend()
    plt.show()
    plt.close()
