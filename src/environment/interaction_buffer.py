from collections import deque

import numpy as np
import pandas as pd


class Buffer:
    def __init__(self) -> None:
        self.buffer = deque()

    def add(
        self,
        id: int,
        state: np.array,
        action: int,
        reward: float,
        next_state: np.array,
        prob: float,
    ) -> None:

        data = (id, state, action, reward, next_state, prob)
        self.buffer.append(data)

    def __len__(self):
        return len(self.buffer)

    def get_df(self, columns):

        stacked = np.stack([x for x in self.buffer])
        return pd.DataFrame(stacked, columns=columns)
