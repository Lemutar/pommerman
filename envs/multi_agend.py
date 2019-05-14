
from pommerman.agents import SimpleAgent
from pommerman.configs import ffa_v0_fast_env, team_v0_fast_env
from pommerman.envs.v0 import Pomme
from pommerman.characters import Bomber
from pommerman import utility

from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.tune.registry import register_env

import numpy as np
import gym
from gym import spaces
from monitoring.monitor import Monitor
from agents.agents import BaseLineAgent, NoDoAgent, SuicidalAgent, RandomMoveAgent


import random

def featurize(obs):
    board = obs["board"].astype(np.float32)
    bomb_blast_strength = obs["bomb_blast_strength"].astype(np.float32)
    bomb_life = obs["bomb_life"].astype(np.float32)

    position = utility.make_np_float(obs["position"])
    ammo = utility.make_np_float([obs["ammo"]])
    blast_strength = utility.make_np_float([obs["blast_strength"]])
    can_kick = utility.make_np_float([obs["can_kick"]])

    teammate = utility.make_np_float([obs["teammate"].value])
    enemies = utility.make_np_float([e.value for e in obs["enemies"]])


    return {'boards': np.stack([board, bomb_blast_strength, bomb_life]),
            'states': np.concatenate([position, ammo, blast_strength, can_kick, teammate, enemies]),}


class MultiAgend(MultiAgentEnv):
    def __init__(self):
        super(MultiAgend, self).__init__()
        self.phase = 0
        self.steps = 0
        self.setup()

    def setup(self):
        agents = []
        if self.phase == 0:
            arr= [0,1]
            agents_index = arr.pop()
            op_index = arr.pop()
            self.agents_index = [agents_index]
            self.enemies_agents_index = [op_index]
            config = ffa_v0_fast_env()
            config["env_kwargs"]["num_wood"]  = 2
            config["env_kwargs"]["num_items"]  = 2
            config["env_kwargs"]["num_rigid"]  = 2
            agents.insert(agents_index, BaseLineAgent(config["agent"](agents_index, config["game_type"])))
            agents.insert(op_index, NoDoAgent(config["agent"](op_index, config["game_type"])))
            print(config["env_kwargs"])
            self.env = Pomme(**config["env_kwargs"])
            self.env.seed()

        if self.phase == 1:
            arr= [0,1]
            random.shuffle(arr)
            agents_index = arr.pop()
            op_index = arr.pop()
            self.agents_index = [agents_index]
            self.enemies_agents_index = [op_index]
            config = ffa_v0_fast_env()
            config["env_kwargs"]["num_wood"]  = 2
            config["env_kwargs"]["num_items"]  = 10
            config["env_kwargs"]["num_rigid"]  = 2
            agents.insert(agents_index, BaseLineAgent(config["agent"](agents_index, config["game_type"])))
            agents.insert(op_index, NoDoAgent(config["agent"](op_index, config["game_type"])))
            print(config["env_kwargs"])
            self.env = Pomme(**config["env_kwargs"])
            self.env.seed()

        if self.phase == 2:
            arr= [0,1]
            random.shuffle(arr)
            agents_index = arr.pop()
            op_index = arr.pop()
            self.agents_index = [agents_index]
            self.enemies_agents_index = [op_index]
            config = ffa_v0_fast_env()
            config["env_kwargs"]["num_wood"]  = 2
            config["env_kwargs"]["num_items"]  = 10
            config["env_kwargs"]["num_rigid"]  = 2
            agents.insert(agents_index, BaseLineAgent(config["agent"](agents_index, config["game_type"])))
            agents.insert(op_index, RandomMoveAgent(config["agent"](op_index, config["game_type"])))
            print(config["env_kwargs"])
            self.env = Pomme(**config["env_kwargs"])
            self.env.seed()

        if self.phase == 3:
            arr= [0,1]
            random.shuffle(arr)
            agents_index = arr.pop()
            op_index = arr.pop()
            self.agents_index = [agents_index]
            self.enemies_agents_index = [op_index]
            config = ffa_v0_fast_env()
            config["env_kwargs"]["num_wood"]  = 2
            config["env_kwargs"]["num_items"]  = 10
            config["env_kwargs"]["num_rigid"]  = 2
            agents.insert(agents_index, BaseLineAgent(config["agent"](agents_index, config["game_type"])))
            agents.insert(op_index, SimpleAgent(config["agent"](op_index, config["game_type"])))
            print(config["env_kwargs"])
            self.env = Pomme(**config["env_kwargs"])
            self.env.seed()

        if self.phase == 4:
            self.agents_index = [0,2]
            self.enemies_agents_index = [1,3]
            config = team_v0_fast_env()
            agents.insert(0, BaseLineAgent(config["agent"](0, config["game_type"])))
            agents.insert(1, NoDoAgent(config["agent"](1, config["game_type"])))
            agents.insert(2, BaseLineAgent(config["agent"](2, config["game_type"])))
            agents.insert(3, NoDoAgent(config["agent"](3, config["game_type"])))
            print(config["env_kwargs"])
            self.env = Pomme(**config["env_kwargs"])
            self.env.seed()

        self.agents_test = agents
        self.env.set_agents(agents)
        self.env.set_init_game_state(None)
        self.observation_space = spaces.Dict({'boards': spaces.Box(low=-1, high=25, shape=(3, 11,11), dtype=np.float32),
                                              'states': spaces.Box(low=-1, high=25, shape=(9,), dtype=np.float32),})

        self.action_space = self.env.action_space
        self.env.reset()

    def set_phase(self, phase):
        print("learn phase " + str(phase))
        self.phase = phase
        self.setup()

    def close(self):
        self.env.close()

    def step(self, actions):
        print(self.env._agents)
        self.steps = self.steps + 1
        obs = self.env.get_observations()
        all_actions = self.env.act(obs)
        assert(len(all_actions) == len(self.agents_index) + len(self.enemies_agents_index))
        for index in self.agents_index :
            try:
                action = actions[index]
            except:
                print(actions)
                action = 0
            assert(all_actions[index] == None)
            all_actions[index] = action

        step_obs = self.env.step(all_actions)
        obs, rew, done, info = {}, {}, {}, {}
        for i in actions.keys():
            obs[i], rew[i], done[i], info[i] = [featurize(step_obs[0][i]),
                                                step_obs[1][i],
                                                step_obs[1][i] == -1 or step_obs[2],
                                                step_obs[3]]

        done["__all__"] = step_obs[2]
        return obs, rew, done, info

    def reset(self):
        self.steps = 0
        obs = self.env.reset()
        return {i: featurize(obs[i]) for i in self.agents_index }


register_env("pommber_team", lambda _:  Monitor(MultiAgend()))
