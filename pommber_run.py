import sys
import ray

from ray.rllib.agents.ppo import PPOAgent
from ray import tune

from envs.multi_agend import MultiAgend
import models.pommberman
import matplotlib.pyplot as plt


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
        name="pommber_cm_46",
        checkpoint_freq=10,
        local_dir="./results",
        resume=True,
        config={
            "num_workers": 2,
            "lr": 5e-4,
            "num_envs_per_worker": 10,
            "observation_filter": "MeanStdFilter",
            "batch_mode": "complete_episodes",
            "train_batch_size": 16000,
            "sgd_minibatch_size": 500,
            "entropy_coeff": 0.01,
            "lambda": 0.95,
            "model": {
                 "custom_model": "pommberman"},
            "env": "pommber_team",
            "callbacks": {
                "on_train_result": tune.function(on_train_result),
            },
        },
    )

run()
#env = MultiAgend()
#env.set_phase(1)
#print(env.reset())

#f, x = plt.subplots(1,8,figsize=(16, 2))
#for i, board in enumerate(env.step({1:0})[0][1]['boards']):
#    x[i].imshow(board)
#    x[i].axis("off")
#plt.show()