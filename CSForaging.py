from typing import Optional, Any

import gymnasium as gym
import numpy as np
import copy

from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter


def generate_space(size=36):
    full = []
    for i in range(5):
        arr = np.zeros((size*3, size*3))
        ind = np.random.randint(0, size, size=2) + size
        arr[ind[0], ind[1]] = 1
        cov = np.random.random(size=(2,))
        cov = cov * 8 + 3  # make singular
        arr = gaussian_filter(arr, sigma=cov)
        circ_arr = np.zeros((size, size))
        # make circular
        for i in range(3):
            for j in range(3):
                circ_arr += arr[i*size:(i+1)*size, j*size:(j+1)*size]
        arr = arr / np.max(arr) # largest value should be 1
        full.append(circ_arr)
    full = np.sum(np.stack(full), axis=0)
    return full


class CategoricalSpace(gym.spaces.Space):

    def __init__(self, prob):
        super().__init__()
        self.space = np.meshgrid(*[np.arange(s) for s in prob.shape])
        self.space[0] = self.space[0].flatten()
        self.space[1] = self.space[1].flatten()
        self.prob_dist = prob.flatten()

    def sample(self):
        ind = np.random.choice(np.arange(len(self.space[0])), p=self.prob_dist)
        return self.space[0][ind], self.space[1][ind]

class ColorShapeForage(gym.Env):
    def __init__(self, reward_param_file=None, frequency_param_file=None, trials=1000):
        # initialize a circle of coordinates in color and shape space.
        _sin = np.sin(2 * np.pi * np.arange(36) / (36))
        _cos = np.cos(2 * np.pi * np.arange(36) / (36))
        _default_stim = np.stack([_sin, _cos], axis=1)
        if reward_param_file is None:
            reward_dist = generate_space(36)
            # treshold
            mins = np.min(reward_dist)
            maxs = np.max(reward_dist)
            step = (maxs - mins) / 4
            for i in range(4):
                reward_dist[np.logical_and(((step * i + mins) <= reward_dist), (reward_dist <= (step * (i+1) + mins)))] = i - 1
            reward_dist = reward_dist.astype(int)
            self.reward_dist = reward_dist
        if frequency_param_file is None:
            # probabilty of shoing an item
            freq_dist = generate_space(36)
            freq_dist = freq_dist / np.sum(freq_dist)
            self.freq_dist = freq_dist
        # observations come from 4 dim shape and color space.
        self.observation_space = CategoricalSpace(prob=self.freq_dist)
        # can choose any of 4 locations.
        self.action_space = gym.spaces.Discrete(4)
        # define default coords
        self.coords = _default_stim
        # track reward values of last observations
        self.rewards = None
        # number of trials per episode before termination
        self.trials = trials
        self.count = 0

    def _get_obs(self):
        # generate observation
        reward_vals = []
        all_obs = []
        for i in range(4):
            o = self.observation_space.sample()
            cval = self.coords[o[0]]
            sval = self.coords[o[1]]
            reward = self.reward_dist[o[0], o[1]]
            obs = np.concatenate([cval, sval], axis=0)
            all_obs.append(obs)
            reward_vals.append(reward)
        reward_vals = np.stack(reward_vals)
        all_obs = np.concatenate(all_obs)
        return all_obs, reward_vals

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        obs, rewards = self._get_obs()
        self.rewards = rewards
        self.count += 1
        return obs, {}

    def step(self, action: int) -> tuple[np.ndarray[float], float, bool, bool, dict[str, Any]]:
        # take action and get reward
        assert self.action_space.contains(action)
        reward = self.rewards[action]
        self.count += 1
        # compute next observation
        obs, self.rewards = self._get_obs()
        terminate = False
        truncate = self.count >= self.trials
        return obs, reward, terminate, truncate, {}

    def plot_reward_space(self):
        # reward space
        plt.imshow(self.reward_dist)
        plt.show()

    def plot_freq_space(self):
        plt.imshow(self.freq_dist)
        plt.show()








