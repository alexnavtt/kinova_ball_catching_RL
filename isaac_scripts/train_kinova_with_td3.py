from matplotlib import pyplot as plt

import numpy as np
from omni.isaac.gym.vec_env import VecEnvBase
env = VecEnvBase(headless=True)

from kinova_task import KinovaTask
task = KinovaTask(name="Kinova")
env.set_task(task, backend="torch")

from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

# The noise objects for TD3
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

params = {
    "policy": 'MlpPolicy',
    "action_noise": action_noise,
    "gamma": 0.98,
    "buffer_size": 200000,
    "learning_starts": 10000,
    "gradient_steps": 1,
    "train_freq": 1,
    "learning_rate": 1e-3,
    "policy_kwargs": {"net_arch": [400, 300]},
    "tensorboard_log": './kinova_tensorboard',
    "verbose": 1
}


model = TD3(
    params["policy"],
    env,
    action_noise=params["action_noise"],
    gamma=params["gamma"],
    buffer_size=params["buffer_size"],
    learning_starts=params["learning_starts"],
    gradient_steps=params["gradient_steps"],
    train_freq=params["train_freq"],
    learning_rate=params["learning_rate"],
    verbose=params["verbose"],
    tensorboard_log=params["tensorboard_log"],
    policy_kwargs=params["policy_kwargs"]
)

model.learn(total_timesteps=100000, log_interval=10)
model.save("kinova")

plt.plot(task._reward_over_time)
plt.savefig("reward.png")

env.close()
