
from pommerman.agents import SimpleAgent
from pommerman.configs import ffa_v0_fast_env, team_v0_fast_env
from pommerman.envs.v0 import Pomme
from pommerman.characters import Bomber
from pommerman import utility

from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.tune.registry import register_env

import numpy as np
import numpy.ma as ma
import gym
from gym import spaces
from monitoring.monitor import Monitor
from agents.agents import BaseLineAgent, NoDoAgent, SuicidalAgent, RandomMoveAgent
from pommerman.constants import Item

import random

def featurize(obs, enemies_agents_index):

    enemies = []
    for agent_id in enemies_agents_index:
        if agent_id == 0:
            enemies.append(Item.Agent0)
        if agent_id == 1:
            enemies.append(Item.Agent1)
        if agent_id == 2:
            enemies.append(Item.Agent2)
        if agent_id == 3:
            enemies.append(Item.Agent3)

    for enemie in obs["enemies"]:
        if enemie not in enemies:
            obs["board"] = ma.masked_equal(obs["board"], enemie.value).filled(fill_value=0)

    board = np.copy(obs["board"])
    board[obs["position"][0], obs["position"][1]] = 0.0
    enemie_pos = np.full((11, 11), 0)
    for enemie in obs["enemies"]:
        enemie_pos = enemie_pos | ma.masked_not_equal(board, enemie.value).filled(fill_value=0)
        board = ma.masked_equal(board, enemie.value).filled(fill_value=0)

    enemie_pos = (enemie_pos > 0).astype(np.float32)

    teammate_pos = np.full((11, 11), 0)

    teammate_pos = ma.masked_not_equal(board, obs["teammate"].value).filled(fill_value=0)
    teammate_pos = (teammate_pos > 0).astype(np.float32)
    board = ma.masked_equal(board, obs["teammate"].value).filled(fill_value=0)
    board = board.astype(np.float32)

    pos = np.full((11, 11), 0)
    pos[obs["position"][0], obs["position"][1]] = 1.0
    pos = pos.astype(np.float32)


    bomb_blast_strength = obs["bomb_blast_strength"].astype(np.float32)
    bomb_life = obs["bomb_life"].astype(np.float32)


    ammo = utility.make_np_float([obs["ammo"]])
    blast_strength = utility.make_np_float([obs["blast_strength"]])
    can_kick = utility.make_np_float([obs["can_kick"]])


    return {'boards': np.stack([board, pos, enemie_pos,teammate_pos, bomb_blast_strength, bomb_life]),
            'states': np.concatenate([ammo, blast_strength, can_kick]),}


class MultiAgend(MultiAgentEnv):
    def __init__(self):
        super(MultiAgend, self).__init__()
        self.phase = 0
        self.next_phase = 0
        self.steps = 0
        self.setup()

    def setup(self):
        agents = []
        if self.phase == 0:
            arr= [0,1]
            random.shuffle(arr)
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
            config["env_kwargs"]["num_wood"]  = 10
            config["env_kwargs"]["num_items"]  = 2
            config["env_kwargs"]["num_rigid"]  = 2
            agents.insert(agents_index, BaseLineAgent(config["agent"](agents_index, config["game_type"])))
            agents.insert(op_index, NoDoAgent(config["agent"](op_index, config["game_type"])))
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
            config["env_kwargs"]["num_wood"]  = 14
            config["env_kwargs"]["num_items"]  = 2
            config["env_kwargs"]["num_rigid"]  = 2
            agents.insert(agents_index, BaseLineAgent(config["agent"](agents_index, config["game_type"])))
            agents.insert(op_index, NoDoAgent(config["agent"](op_index, config["game_type"])))
            self.env = Pomme(**config["env_kwargs"])
            print(config["env_kwargs"])
            self.env.seed()

        if self.phase == 3:
            arr= [0,1]
            random.shuffle(arr)
            agents_index = arr.pop()
            op_index = arr.pop()
            self.agents_index = [agents_index]
            self.enemies_agents_index = [op_index]
            config = ffa_v0_fast_env()
            config["env_kwargs"]["num_wood"]  = 16
            config["env_kwargs"]["num_items"]  = 2
            config["env_kwargs"]["num_rigid"]  = 2
            agents.insert(agents_index, BaseLineAgent(config["agent"](agents_index, config["game_type"])))
            agents.insert(op_index, NoDoAgent(config["agent"](op_index, config["game_type"])))
            self.env = Pomme(**config["env_kwargs"])
            print(config["env_kwargs"])
            self.env.seed()

        if self.phase == 4:
            arr= [0,1]
            random.shuffle(arr)
            agents_index = arr.pop()
            op_index = arr.pop()
            self.agents_index = [agents_index]
            self.enemies_agents_index = [op_index]
            config = ffa_v0_fast_env()
            config["env_kwargs"]["num_wood"]  = 18
            config["env_kwargs"]["num_items"]  = 2
            config["env_kwargs"]["num_rigid"]  = 2
            agents.insert(agents_index, BaseLineAgent(config["agent"](agents_index, config["game_type"])))
            agents.insert(op_index, NoDoAgent(config["agent"](op_index, config["game_type"])))
            self.env = Pomme(**config["env_kwargs"])
            print(config["env_kwargs"])
            self.env.seed()

        if self.phase == 5:
            arr= [0,1]
            random.shuffle(arr)
            agents_index = arr.pop()
            op_index = arr.pop()
            self.agents_index = [agents_index]
            self.enemies_agents_index = [op_index]
            config = ffa_v0_fast_env()
            config["env_kwargs"]["num_wood"]  = 20
            config["env_kwargs"]["num_items"]  = 2
            config["env_kwargs"]["num_rigid"]  = 2
            agents.insert(agents_index, BaseLineAgent(config["agent"](agents_index, config["game_type"])))
            agents.insert(op_index, NoDoAgent(config["agent"](op_index, config["game_type"])))
            self.env = Pomme(**config["env_kwargs"])
            print(config["env_kwargs"])
            self.env.seed()

        self.agents_test = agents
        self.env.set_agents(agents)
        self.env.set_init_game_state(None)
        self.observation_space = spaces.Dict({'boards': spaces.Box(low=-1, high=25, shape=(6, 11,11), dtype=np.float32),
                                              'states': spaces.Box(low=-1, high=25, shape=(3,), dtype=np.float32),})

        self.action_space = self.env.action_space
        self.env.reset()

    def set_phase(self, phase):
        print("learn phase " + str(phase))
        self.next_phase = phase

    def close(self):
        self.env.close()

    def step(self, actions):
        self.steps = self.steps + 1
        obs = self.env.get_observations()
        all_actions = self.env.act(obs)
        assert(len(all_actions) == len(self.agents_index) + len(self.enemies_agents_index))


        for index in self.agents_index :
            try:
                action = actions[index]
            except:
                print("WWWRRROOOONNNNG")
                action = 0
            assert(all_actions[index] == None)
            all_actions[index] = action

        step_obs = self.env.step(all_actions)
        obs, rew, done, info = {}, {}, {}, {}
        for i in actions.keys():
            obs[i], rew[i], done[i], info[i] = [featurize(step_obs[0][i], self.enemies_agents_index ),
                                                step_obs[1][i],
                                                step_obs[1][i] == -1 or step_obs[2],
                                                step_obs[3]]

        done["__all__"] = step_obs[2]
        return obs, rew, done, info

    def reset(self):
        self.steps = 0
        self.phase = self.next_phase
        self.setup()
        obs = self.env.reset()
        return {i: featurize(obs[i], self.enemies_agents_index) for i in self.agents_index }


register_env("pommber_team", lambda _:  MultiAgend())
