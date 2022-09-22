import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from configs.config import CFG_DICT


def get_cvae_dataloader(dataset):
    train = CVAEDataset(dataset, is_train=True)
    val = CVAEDataset(dataset, is_train=False)

    train_loader = DataLoader(
        train,
        batch_size=CFG_DICT["TRAIN"]["BATCHSIZE"],
        shuffle=True,
        num_workers=CFG_DICT["TRAIN"]["WORKER"],
    )
    test_loader = DataLoader(
        val,
        batch_size=CFG_DICT["TRAIN"]["EVAL_BATCHSIZE"],
        shuffle=False,
        num_workers=CFG_DICT["TRAIN"]["WORKER"],
    )
    return train_loader, test_loader


class CVAEDataset(Dataset):

    """
    note : torch.utils.data.Datasetを継承した、データの読み込み、データの前処理を行うクラス
    """

    def __init__(self, df: pd.DataFrame, train_size=0.8, is_train=True):

        self.df = df
        self.train_size = train_size
        self.train, self.test = self._split_data()

        if is_train:
            self.dataset = self.train
        else:
            self.dataset = self.test

    def _split_data(self):
        n_train = int(len(self.df) * self.train_size)
        train = self.df.sample(n=n_train, random_state=1)
        test = self.df[~self.df.index.isin(train.index)]
        return train, test

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):

        series = self.dataset.iloc[index, :]

        X = np.zeros((int(np.max(self.df.Action) + 1)))
        X[int(series.Action)] = 1.0

        Y = np.zeros((3))
        Y[0] = series.State / 50
        Y[1] = series.Reward / 25
        Y[2] = series.Next_state / 50

        X = torch.from_numpy(X.astype(np.float32)).clone()
        Y = torch.from_numpy(Y.astype(np.float32)).clone()

        return X, Y
