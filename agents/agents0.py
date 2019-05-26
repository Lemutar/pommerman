import numpy as np
from pommerman.agents import BaseAgent

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten

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

class CuriosityAgent(BaseAgent):
    def act(self, obs, action_space):
        pass

    def get_reward(self, obs, curiosity):
        curiosity.step(obs)



class Curiosity():
    def __init__(self, env):
        self.fix_model = Sequential()
        self.fix_model.add(Conv2D(32, kernel_size=1, activation='relu',kernel_initializer='random_uniform', input_shape=(11,11, 12)))
        self.fix_model.add(Flatten())
        self.fix_model.add(Dense(128, activation='softmax', kernel_initializer='random_uniform'))
        self.fix_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

        self.train_model = Sequential()
        self.train_model.add(Conv2D(32, kernel_size=3, activation='relu', input_shape=(11,11, 12)))
        self.train_model.add(Flatten())
        self.train_model.add(Dense(128, activation='softmax'))
        self.train_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
        self.obs_batch=[]
        self.result_batch=[]

    def step(self, obs):
        self.obs_batch.append(obs['boards'])
        c_obs = np.expand_dims(obs['boards'], axis=0)
        rand_result = self.fix_model.predict(c_obs)
        self.result_batch.append(rand_result)
        mse = (np.square(rand_result - self.train_model.predict(c_obs))).mean(axis=None)
        return mse * 1000

    def train(self):
        if len(self.obs_batch) > 0:
            obs = np.stack(self.obs_batch)
            res = np.concatenate(self.result_batch)
            print(obs.shape)
            print(res.shape)
            self.train_model.fit(obs, res, epochs=3, verbose=0)
            self.obs_batch=[]
            self.result_batch=[]
