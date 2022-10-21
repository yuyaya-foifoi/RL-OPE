import gym
import numpy as np


class ACControl(gym.Env):
    def __init__(self):
        self.ACTION_NUM = 7
        self.action_space = gym.spaces.Discrete(self.ACTION_NUM)

        self.STATE_NUM = 50
        self.observation_space = gym.spaces.Discrete(self.STATE_NUM)

        self.best_tmp = 25
        self.tmp = 5  # np.random.choice(self.STATE_NUM, 1).item()

    def sigmoid(self, a):
        return 1 / (1 + np.exp(-a))

    def reset(self):
        return self.tmp

    def _calc_change(self, action_idx):
        return action_idx - np.median(np.arange(self.ACTION_NUM))

    def step(self, action_idx):
        change = self._calc_change(action_idx)
        self.tmp += change
        return self.tmp, self.sigmoid(-np.abs(self.best_tmp - self.tmp)) / 0.5

    def render(self):
        pass
