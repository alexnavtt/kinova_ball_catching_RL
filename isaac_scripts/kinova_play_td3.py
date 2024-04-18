# create isaac environment
from omni.isaac.gym.vec_env import VecEnvBase
env = VecEnvBase(headless=False)

# create task and register task
from kinova_task import KinovaTask
task = KinovaTask(name="Kinova")
env.set_task(task, backend="torch")

# import stable baselines
from stable_baselines3 import TD3

# Run inference on the trained policy
model = TD3.load("kinova")
env._world.reset()
obs, _ = env.reset()
while env._simulation_app.is_running():
    action, _states = model.predict(obs)
    obs, rewards, terminated, truncated, info = env.step(action)

env.close()
