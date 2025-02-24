import gym
import torch

import numpy as np

class GymExperimentEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, data_dim, data_lims=[-100, 100]):
        """Initialize a Xd taskspace.
        """
        self.dt = 1e0

        self.min_dim = np.array([data_lims[0]] * data_dim)
        self.max_dim = np.array([data_lims[1]] * data_dim)
        self.action_space = gym.spaces.Box(self.max_dim, self.max_dim)
        self.observation_space = gym.spaces.Box(self.max_dim, self.max_dim)

        self.reset()

    def step(self, action):
        """ Apply an action to the environment.

        Args:
            action (Array Like): Action taken.

        Returns:
            np.ndarray, int, bool, Dict: Observations and environment states.
        """
        vector = torch.from_numpy(np.asarray(action)).float()
        vector = torch.nn.functional.normalize(vector, 2, 0)
        vector = vector.cpu().detach().numpy()

        self.q = self.q  + vector * self.dt
        self.q = np.clip(self.q, self.min_dim, self.max_dim)

        observation = np.copy(self.q)
        return observation, 0, False, {}

    def reset(self):
        """ Reset the environment """
        self.q = self.min_dim + np.random.random(self.min_dim.shape[0]) * (self.max_dim - self.min_dim)

        observation = np.copy(self.q)
        return observation

    def render(self, mode='human'):
        """ Not needed at this point """
        pass

    def close(self):
        " Not needed at this point "
        pass
