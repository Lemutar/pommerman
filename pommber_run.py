import os
import sys

import pickle
import resource
import sys
import ray



from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.tune.registry import register_env
from ray.rllib.agents.ppo import PPOAgent
from ray import tune
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers import xavier_initializer

from ray.rllib.models import FullyConnectedNetwork, Model, ModelCatalog
from gym.spaces import Discrete, Box


from envs.multi_agend import MultiAgend
from models.pommberman import Pommberman
import cv2




class PhasePPO(PPOAgent):

    def __init__(self, config=None, env=None, logger_creator=None):
        super(PhasePPO, self).__init__(config=config,env=env, logger_creator=logger_creator)
        self.train_phase = 0

def on_episode_end(info):
    env = info["env"]
    episode.custom_metrics["train_phase"] = env.get_phase()

def on_train_result(info):
    trainer = info["trainer"]
    result = info["result"]
    if result["episode_reward_mean"] > 0.95 and trainer.train_phase == 0:
        trainer.train_phase  = 1
    elif result["episode_reward_mean"] > 0.95 and trainer.train_phase == 1:
        trainer.train_phase  = 2
    elif result["episode_reward_mean"] > 0.95 and trainer.train_phase == 2:
        trainer.train_phase  = 3
    elif result["episode_reward_mean"] > 0.95 and trainer.train_phase == 3:
        trainer.train_phase  = 4


    phase = trainer.train_phase
    trainer.optimizer.foreach_evaluator(
        lambda ev: ev.foreach_env(
            lambda env: env.set_phase(phase)))


def run():
    sys.setrecursionlimit(1000)
    ray.shutdown()
    ray.init()

    tune.run(
        PhasePPO,
        name="pommber_cm",
        checkpoint_freq=10,
        local_dir="./results",
        config={
            "monitor": True,
            "num_workers": 2,
            "vf_share_layers":True,
            "model": {
                 "custom_model": "Pommberman"},
            "env": "pommber_team",
            "callbacks": {
                "on_train_result": tune.function(on_train_result),
            },
        },
    )

run()
