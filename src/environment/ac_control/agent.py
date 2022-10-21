import numpy as np


class Agent:
    def __init__(self):
        self.cold_dist = [0.05, 0.05, 0.05, 0.05, 0.1, 0.2, 0.5]
        self.quit_cold_dist = [0.05, 0.05, 0.05, 0.1, 0.15, 0.2, 0.4]
        self.bit_cold_dist = [0.05, 0.05, 0.1, 0.1, 0.3, 0.2, 0.2]
        self.bit_hot_dist = [i for i in reversed(self.bit_cold_dist)]
        self.quit_hot_dist = [i for i in reversed(self.quit_cold_dist)]
        self.hot_dist = [i for i in reversed(self.cold_dist)]

    def get_action(self, state):
        dist = self._get_dist(state)
        action = np.random.choice(len(dist), 1, p=dist)
        return action, dist

    def _get_dist(self, state):

        if state in np.arange(0, 10):
            return self.cold_dist

        if state in np.arange(10, 20):
            return self.quit_cold_dist

        if state in np.arange(20, 25):
            return self.bit_cold_dist

        if state in np.arange(25, 30):
            return self.bit_hot_dist

        if state in np.arange(30, 40):
            return self.quit_hot_dist

        if state in np.arange(40, 50):
            return self.hot_dist
