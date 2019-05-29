
from pommerman.agents import SimpleAgent
from pommerman.configs import ffa_v0_fast_env, team_v0_fast_env
from pommerman.envs.v0 import Pomme
from pommerman import utility

from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.tune.registry import register_env

import numpy as np
import numpy.ma as ma
from gym import spaces
from agents.agents import BaseLineAgent, NoDoAgent, SuicidalAgent, RandomMoveAgent, Curiosity
from pommerman.constants import Item, DEFAULT_BOMB_LIFE
from monitoring.monitor import Monitor
import json
import copy

import random


class MultiAgend(MultiAgentEnv):
    def __init__(self):
        super(MultiAgend, self).__init__()
        self.phase = 0
        self.next_phase = 0
        self.steps = 0
        self.last_featurize_obs = None
        self.setup()


    def featurize(self, obs):

        enemies = []
        for agent_id in self.enemies_agents_index :
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

        wood = ma.masked_not_equal(board, 2).filled(fill_value=0)
        wood = (wood > 0).astype(np.float32)
        board = ma.masked_equal(board, 2).filled(fill_value=0)

        stone = ma.masked_not_equal(board, 1).filled(fill_value=0)
        stone = (stone > 0).astype(np.float32)
        board = ma.masked_equal(board, 1).filled(fill_value=0)
        enemie_pos = (enemie_pos > 0).astype(np.float32)

        board = ma.masked_equal(board, obs["teammate"].value).filled(fill_value=0)

        flames = ma.masked_not_equal(board, 4).filled(fill_value=0)
        flames = (flames > 0).astype(np.float32)

        board = ma.masked_equal(board, 4).filled(fill_value=0)
        board = ma.masked_equal(board, 3).filled(fill_value=0)

        teammate_pos = ma.masked_not_equal(board, obs["teammate"].value).filled(fill_value=0)
        teammate_pos = (teammate_pos > 0).astype(np.float32)
        board = ma.masked_equal(board, obs["teammate"].value).filled(fill_value=0)
        items = board.astype(np.float32)

        pos = np.full((11, 11), 0)
        pos[obs["position"][0], obs["position"][1]] = 1.0
        pos = pos.astype(np.float32)

        bomb_life = obs["bomb_life"].astype(np.float32)
        bomb_blast_strength = obs["bomb_blast_strength"].astype(np.float32)

        ammo = utility.make_np_float([obs["ammo"]])
        blast_strength = utility.make_np_float([obs["blast_strength"]])
        can_kick = utility.make_np_float([obs["can_kick"]])
        game_end = utility.make_np_float([(self.max_steps - self.steps) / self.max_steps])

        actual_featurize_obs = {'boards': np.stack([
                                    enemie_pos,
                                    pos,
                                    wood,
                                    stone,
                                    items,
                                    flames,
                                    teammate_pos,
                                    bomb_life,
                                    bomb_blast_strength], axis=0),
                 'states': np.concatenate([ammo, blast_strength, can_kick, game_end]),}


        if self.last_featurize_obs == None:
                featurize_obs =  {'boards': np.concatenate([actual_featurize_obs['boards'],actual_featurize_obs['boards']],axis=0),
                                  'states': np.concatenate([actual_featurize_obs['states'],actual_featurize_obs['states']]),}
        else:
                featurize_obs =  {'boards': np.concatenate([self.last_featurize_obs['boards'],actual_featurize_obs['boards']],axis=0),
                                  'states': np.concatenate([self.last_featurize_obs['states'],actual_featurize_obs['states']]),}

        self.last_featurize_obs = actual_featurize_obs
        return featurize_obs


    def setup(self):
        agents = []
        if self.phase == 0:
            arr= [0, 1]
            random.shuffle(arr)
            agents_index = arr.pop()
            op_index = arr.pop()
            self.agents_index = [agents_index]
            self.enemies_agents_index = [op_index]
            self.max_steps = 200
            config = ffa_v0_fast_env()
            config["env_kwargs"]["max_steps"] = self.max_steps
            agents.insert(agents_index, BaseLineAgent(config["agent"](agents_index, config["game_type"])))
            agents.insert(op_index, NoDoAgent(config["agent"](op_index, config["game_type"])))
            self.env = Pomme(**config["env_kwargs"])
            self.env.set_agents(agents)
            init_state = {'board_size': '11', 'step_count': '0', 'board': '','agents': '[{"agent_id": 0, "is_alive": true, "position": [1, 1], "ammo": 1, "blast_strength": 2, "can_kick": false}, {"agent_id": 1, "is_alive": true, "position": [9, 0], "ammo": 1, "blast_strength": 2, "can_kick": false}]', 'bombs': '[]', 'flames': '[]', 'items': '[]', 'intended_actions': '[0, 0]'}
            board = np.full((11, 11), 0)
            init_state['board'] = json.dumps(board.tolist())
            agents_json = json.loads(copy.copy(init_state['agents']))
            random_pos = np.random.choice(board.shape[0], (2, 2), replace=False)
            agents_json[0]["position"] = random_pos[0].tolist()
            agents_json[1]["position"] = random_pos[1].tolist()
            init_state['agents'] = json.dumps(agents_json)
            self.env._init_game_state = init_state
            self.env.reset()

        if self.phase == 1:
            arr= [0, 1]
            random.shuffle(arr)
            agents_index = arr.pop()
            op_index = arr.pop()
            self.agents_index = [agents_index]
            self.enemies_agents_index = [op_index]
            self.max_steps = 200
            config = ffa_v0_fast_env()
            config["env_kwargs"]["max_steps"] = self.max_steps
            agents.insert(agents_index, BaseLineAgent(config["agent"](agents_index, config["game_type"])))
            agents.insert(op_index, NoDoAgent(config["agent"](op_index, config["game_type"])))
            self.env = Pomme(**config["env_kwargs"])
            self.env.set_agents(agents)
            init_state = {'board_size': '11', 'step_count': '0', 'board': '','agents': '[{"agent_id": 0, "is_alive": true, "position": [1, 1], "ammo": 1, "blast_strength": 2, "can_kick": false}, {"agent_id": 1, "is_alive": true, "position": [9, 0], "ammo": 1, "blast_strength": 2, "can_kick": false}]', 'bombs': '[]', 'flames': '[]', 'items': '[]', 'intended_actions': '[0, 0]'}
            board = np.full((11, 11), 0)
            board[5,:] = (np.ones(11) * 2)
            agents_json = json.loads(copy.copy(init_state['agents']))
            agents_json[0]["position"] = [random.randint(0, 4), random.randint(0, 10)]
            agents_json[1]["position"] = [random.randint(6, 10), random.randint(0, 10)]
            init_state['agents'] = json.dumps(agents_json)
            init_state['board'] = json.dumps(board.tolist())
            self.env._init_game_state = init_state
            self.env.reset()

        self.observation_space = spaces.Dict({'boards': spaces.Box(low=-1, high=25, shape=(11, 11, 18), dtype=np.float32),
                                              'states': spaces.Box(low=-1, high=25, shape=(8,), dtype=np.float32)})

        self.action_space = self.env.action_space


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
                action = 0
            assert(all_actions[index] == None)
            all_actions[index] = action

        step_obs = self.env.step(all_actions)
        obs, rew, done, info = {}, {}, {}, {}
        for i in actions.keys():
            obs[i], rew[i], done[i], info[i] = [self.featurize(step_obs[0][i]),
                                                step_obs[1][i],
                                                step_obs[1][i] == -1 or step_obs[2],
                                                step_obs[3]]

        done["__all__"] = step_obs[2]
        return obs, rew, done, info


    def reset(self):
        self.steps = 0
        self.phase = self.next_phase
        self.setup()
        obs = self.env.get_observations()
        return {i: self.featurize(obs[i]) for i in self.agents_index }


register_env("pommber_team", lambda _:  MultiAgend())
