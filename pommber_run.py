import sys
import ray

from ray.rllib.agents.ppo import PPOAgent
from ray import tune

from envs.multi_agend import MultiAgend

import models.pommberman_lstm
import models.pommberman
import matplotlib.pyplot as plt
from agents.agents import BaseLineAgent, NoDoAgent, SuicidalAgent, RandomMoveAgent, Curiosity


class PhasePPO(PPOAgent):

    def __init__(self, config=None, env=None, logger_creator=None):
        super(PhasePPO, self).__init__(config=config,env=env, logger_creator=logger_creator)
        self.train_phase = 0


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
    elif result["episode_reward_mean"] > 0.95 and trainer.train_phase == 4:
        trainer.train_phase  = 5
    elif result["episode_reward_mean"] > 0.95 and trainer.train_phase == 5:
        trainer.train_phase  = 6

    phase = trainer.train_phase
    trainer.optimizer.foreach_evaluator(
        lambda ev: ev.foreach_env(
            lambda env: env.set_phase(phase)))


def run():
    sys.setrecursionlimit(56000)
    ray.shutdown()
    ray.init(num_gpus=1)
    tune.run(
        PhasePPO,
        name="pommber_cm_lstm_111",
        checkpoint_freq=10,
        local_dir="./results",
        config={
            "num_workers": 15,
            "num_gpus": 1,
            "num_envs_per_worker": 16,
            "observation_filter": "MeanStdFilter",
            "batch_mode": "complete_episodes",
            "train_batch_size": 16000,
            "sgd_minibatch_size": 1600,
            "vf_share_layers": True,
            "lr": 5e-4,
            "gamma": 0.997,
            "model": {
                 "use_lstm": True,
                 "max_seq_len": 60,
                 "lstm_cell_size": 128,
                 "custom_model": "pommberman"},
            "env": "pommber_team",
            "callbacks": {
                "on_train_result": tune.function(on_train_result),
            },
        },
    )

def test():
    env = MultiAgend()
    env.set_phase(0)
    p = env.agents_index[0]

    env.step({p:1})
    f, x = plt.subplots(1,18,figsize=(36, 2))
    for i, board in enumerate(env.step({p:1})[0][p]['boards']):
        x[i].imshow(board)
        x[i].axis("off")
    plt.show()

test()
