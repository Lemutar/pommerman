import numpy as np
from pommerman.agents import BaseAgent

class BaseLineAgent(BaseAgent):
    def act(self, obs, action_space):
        pass

class NoDoAgent(BaseAgent):
    def act(self, obs, action_space):
        return 0

class SuicidalAgent(BaseAgent):
    def act(self, obs, action_space):
        return 5

class RandomMoveAgent(BaseAgent):
    def act(self, obs, action_space):
        return np.random.choice([1,2,3,4])
