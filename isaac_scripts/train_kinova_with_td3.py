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

model = TD3("MlpPolicy", env, action_noise=action_noise, verbose=1)
model.learn(total_timesteps=10000, log_interval=10)
model.save("kinova")

env.close()
