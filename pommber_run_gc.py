import sys
import ray

from ray.rllib.agents.ppo import PPOAgent
from ray import tune

from envs.multi_agend_1 import MultiAgend
import models.pommberman_lstm
import models.pommberman


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


    phase = trainer.train_phase
    trainer.optimizer.foreach_evaluator(
        lambda ev: ev.foreach_env(
            lambda env: env.set_phase(phase)))


def run():
    sys.setrecursionlimit(1000)
    ray.shutdown()
    ray.init(redis_address="localhost:6379")

    tune.run(
        PhasePPO,
        name="pommber_cm",
        checkpoint_freq=10,
        local_dir="./results",
        config={
            "num_workers": 16,
            "vf_share_layers":True,
            "model": {
                 "custom_model": "pommberman_lstm"},
            "env": "pommber_team",
            "callbacks": {
                "on_train_result": tune.function(on_train_result),
            },
        },
    )

run()
#env = MultiAgend()
#print(env.reset())
#print(env.step({1:1}))
