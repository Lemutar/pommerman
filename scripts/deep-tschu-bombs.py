import os
import sys
import numpy as np
import gym
from gym import spaces
import pickle
import resource
import ray
import click

from pommerman.agents import SimpleAgent, RandomAgent, PlayerAgent, BaseAgent
from pommerman.configs import ffa_v0_fast_env, team_v0_fast_env
from pommerman.envs.v0 import Pomme
from pommerman.characters import Bomber
from pommerman import utility

from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.tune.registry import register_env
from ray.rllib.agents.ppo import PPOAgent
from ray import tune

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers import xavier_initializer

from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.models.misc import normc_initializer
from ray.rllib.models.model import Model

from ray.rllib.models import FullyConnectedNetwork, Model, ModelCatalog
from gym.spaces import Discrete, Box


class PommermanModel(Model):
    def _build_layers_v2(self, input_dict, num_outputs, options):

        # Parse options
        inputs = input_dict["obs"]
        convs = [[16, [2, 2], 4], [32, [2, 2], 3], [32, [2, 2], 2], [128, [1, 1], 1]]
        hiddens = [128, 128]
        fcnet_activation = options.get("fcnet_activation", "tanh")
        if fcnet_activation == "tanh":
            activation = tf.nn.tanh
        elif fcnet_activation == "relu":
            activation = tf.nn.relu

        vision_in = inputs["boards"]
        metrics_in = inputs["states"]

        # Setup vision layers
        with tf.name_scope("pommer_vision"):
            for i, (out_size, kernel, stride) in enumerate(convs[:-1], 1):
                vision_in = slim.conv2d(
                    vision_in, out_size, kernel, stride, scope="conv{}".format(i)
                )
            out_size, kernel, stride = convs[-1]
            vision_in = slim.conv2d(
                vision_in, out_size, kernel, stride, padding="VALID", scope="conv_out"
            )
            vision_in = tf.squeeze(vision_in, [1, 2])

        # Setup metrics layer
        with tf.name_scope("pommer_metrics"):
            metrics_in = slim.fully_connected(
                metrics_in,
                64,
                weights_initializer=xavier_initializer(),
                activation_fn=activation,
                scope="metrics_out",
            )

        with tf.name_scope("pommer_out"):
            i = 1
            last_layer = tf.concat([vision_in, metrics_in], axis=1)
            for size in hiddens:
                last_layer = slim.fully_connected(
                    last_layer,
                    size,
                    weights_initializer=xavier_initializer(),
                    activation_fn=activation,
                    scope="fc{}".format(i),
                )
                i += 1
            output = slim.fully_connected(
                last_layer,
                num_outputs,
                weights_initializer=normc_initializer(0.01),
                activation_fn=None,
                scope="fc_out",
            )

        return output, last_layer


ModelCatalog.register_custom_model("PommermanModel1", PommermanModel)


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
        return np.random.choice([1, 2, 3, 4])


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

    return {
        "boards": np.stack([board, bomb_blast_strength, bomb_life]),
        "states": np.concatenate(
            [position, ammo, blast_strength, can_kick, teammate, enemies]
        ),
    }


class MultiAgent(MultiAgentEnv):
    def __init__(self):
        super(MultiAgent, self).__init__()
        self.phase = 0
        self.setup()

    def setup(self):
        agents = []
        if self.phase == 0:
            self.agents_index = [1, 3]
            self.enemies_agents_index = [0, 2]
            config = team_v0_fast_env()
            config["env_kwargs"]["num_wood"] = 2
            config["env_kwargs"]["num_items"] = 2
            config["env_kwargs"]["num_rigid"] = 20
            agents.insert(0, SuicidalAgent(config["agent"](0, config["game_type"])))
            agents.insert(2, NoDoAgent(config["agent"](2, config["game_type"])))
            print(config["env_kwargs"])
            self.env = Pomme(**config["env_kwargs"])
            self.env.seed()

        if self.phase == 1:
            self.agents_index = [1, 3]
            self.enemies_agents_index = [0, 2]
            config = team_v0_fast_env()
            config["env_kwargs"]["num_wood"] = 2
            config["env_kwargs"]["num_items"] = 2
            config["env_kwargs"]["num_rigid"] = 36
            agents.insert(0, SuicidalAgent(config["agent"](0, config["game_type"])))
            agents.insert(2, NoDoAgent(config["agent"](2, config["game_type"])))
            print(config["env_kwargs"])
            self.env = Pomme(**config["env_kwargs"])
            self.env.seed()

        if self.phase == 2:
            self.agents_index = [1, 3]
            self.enemies_agents_index = [0, 2]
            config = team_v0_fast_env()
            config["env_kwargs"]["num_wood"] = 2
            config["env_kwargs"]["num_items"] = 2
            config["env_kwargs"]["num_rigid"] = 36
            agents.insert(0, NoDoAgent(config["agent"](0, config["game_type"])))
            agents.insert(2, NoDoAgent(config["agent"](2, config["game_type"])))
            print(config["env_kwargs"])
            self.env = Pomme(**config["env_kwargs"])
            self.env.seed()

        if self.phase == 3:
            self.agents_index = [1, 3]
            self.enemies_agents_index = [0, 2]
            config = team_v0_fast_env()
            config["env_kwargs"]["num_wood"] = 2
            config["env_kwargs"]["num_items"] = 2
            config["env_kwargs"]["num_rigid"] = 36
            agents.insert(0, NoDoAgent(config["agent"](0, config["game_type"])))
            agents.insert(2, NoDoAgent(config["agent"](2, config["game_type"])))
            print(config["env_kwargs"])
            self.env = Pomme(**config["env_kwargs"])
            self.env.seed()

        if self.phase == 4:
            self.agents_index = [1, 3]
            self.enemies_agents_index = [0, 2]
            config = team_v0_fast_env()
            config["env_kwargs"]["num_wood"] = 0
            config["env_kwargs"]["num_items"] = 10
            config["env_kwargs"]["num_rigid"] = 36
            agents.insert(0, SuicidalAgent(config["agent"](0, config["game_type"])))
            agents.insert(2, SimpleAgent(config["agent"](2, config["game_type"])))
            print(config["env_kwargs"])
            self.env = Pomme(**config["env_kwargs"])
            self.env.seed()

        for agent_id in self.agents_index:
            agents.insert(
                agent_id, BaseLineAgent(config["agent"](agent_id, config["game_type"]))
            )

        self.env.set_agents(agents)
        self.env.set_init_game_state(None)
        self.observation_space = spaces.Dict(
            {
                "boards": spaces.Box(low=-1, high=20, shape=(3, 11, 11)),
                "states": spaces.Box(low=-1, high=20, shape=(9,)),
            }
        )

        spaces.Box(low=-1.0, high=20.0, shape=(372,), dtype=np.float32)
        self.action_space = self.env.action_space

    def set_phase(self, phase):
        print("learn phase " + str(phase))
        self.phase = phase
        self.setup()
        self.reset()

    def step(self, actions):
        obs = self.env.get_observations()
        all_actions = self.env.act(obs)
        for index in self.agents_index:
            try:
                action = actions[index]
            except:
                action = 0
            all_actions[index] = action

        step_obs = self.env.step(all_actions)
        obs, rew, done, info = {}, {}, {}, {}
        for i in actions.keys():
            obs[i], rew[i], done[i], info[i] = [
                featurize(step_obs[0][i]),
                step_obs[1][i],
                step_obs[1][i] == -1 or step_obs[2],
                step_obs[3],
            ]

        done["__all__"] = step_obs[2]
        return obs, rew, done, info

    def reset(self):
        obs = self.env.reset()
        return {i: featurize(obs[i]) for i in self.agents_index}


register_env("pommer_team", lambda _: MultiAgent())


sys.setrecursionlimit(1000)


class PhasePPO(PPOAgent):
    def __init__(self, config=None, env=None, logger_creator=None):
        super(PhasePPO, self).__init__(
            config=config, env=env, logger_creator=logger_creator
        )
        self.train_phase = 0


def on_episode_end(info):
    env = info["env"]
    episode.custom_metrics["train_phase"] = env.get_phase()


def on_train_result(info):
    trainer = info["trainer"]
    result = info["result"]
    if result["episode_reward_mean"] > 1.5 and trainer.train_phase == 0:
        trainer.train_phase = 1
    elif result["episode_reward_mean"] > 1.5 and trainer.train_phase == 1:
        trainer.train_phase = 2
    elif result["episode_reward_mean"] > 1.5 and trainer.train_phase == 2:
        trainer.train_phase = 3
    elif result["episode_reward_mean"] > 1.5 and trainer.train_phase == 3:
        trainer.train_phase = 4

    phase = trainer.train_phase
    trainer.optimizer.foreach_evaluator(
        lambda ev: ev.foreach_env(lambda env: env.set_phase(phase))
    )


ray.init(num_gpus=0)


@click.command()
@click.option("--num_cpus", default=0)
@click.option("--num_gpus", default=8)
def main(num_cpus, num_gpus):
    tune.run(
        PhasePPO,
        name="pommer_cnet_0",
        checkpoint_freq=10,
        local_dir="./results",
        resume=False,
        reuse_actors=True,
        config={
            "num_workers": 2,
            "vf_share_layers": True,
            "model": {"custom_model": "PommermanModel1"},
            "env": "pommer_team",
            "callbacks": {"on_train_result": tune.function(on_train_result)},
        },
        resources_per_trial={"cpu": num_cpus, "gpu": num_gpus},
    )


if __name__ == "__main__":
    main()
