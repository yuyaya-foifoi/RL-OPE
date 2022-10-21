import numpy as np
import pandas as pd
from tqdm import tqdm

from src.environment.interaction_buffer import Buffer


def behavior_policy_interaction(
    env, buffer: Buffer, policy_name: str, columns: list, trial_len: int, model
):

    observation = env.reset()

    for time in range(trial_len):

        action, dist = model.get_action(observation)

        next_observation, reward = env.step(action.item())
        buffer.add(
            policy_name,
            observation,
            action.item(),
            reward,
            next_observation,
            dist[action.item()],
        )
        observation = next_observation

    env.close()

    return buffer.get_df(columns)


def estimate_policy_interaction(
    env, buffer: Buffer, policy_name: str, columns: list, trial_len: int, model
) -> pd.DataFrame:

    observation = env.reset()

    for _ in tqdm(range(trial_len)):

        pi_s = model.predict_proba(
            np.array(observation).reshape(-1, 1)
        ).flatten()
        action = np.random.choice(a=np.arange(len(pi_s)), p=pi_s)

        next_observation, reward = env.step(action)
        buffer.add(
            policy_name,
            observation,
            action,
            reward,
            next_observation,
            pi_s[action],
        )
        observation = next_observation

    env.close()

    return buffer.get_df(columns)
