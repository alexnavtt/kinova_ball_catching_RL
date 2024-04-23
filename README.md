# Ball Catching with Kinova Gen 3 using RL
Reinforcement learning project aimed at having a Kinova Gen3 manipulator catch a ball in the air

## Setup

1. Make sure you have the `OmniIsaacGymEnvs` repository cloned to your device. Follow their installation instructions if not.
2. Create a symbolic link from the directory `OmniIsaacGymEnvs/omniisaacgymenvs/cgf/task/` to `kinova_ball_catching_RL/config/KinovaTask.yaml`
3. Create a symbolic link from the directory `OmniIsaacGymEnvs/omniisaacgymenvs/cgf/train/` to `kinova_ball_catching_RL/config/KinovaTaskPPO.yaml`
4. Navigate to `OmniIsaacGymEnvs/omniisaacgymenvs/utils/task_util.py`.
5. Inside the `import_tasks()` function, add `from kinova_task import KinovaTask`.
6. Inside the `task_map` dictionary, add an entry `"KinovaTask": KinovaTask`.

## Running

1. Add the `isaac_scripts` folder in this repo to your `PYTHONPATH` environment variable manually. Example: `export PYTHONPATH=$PYTHONPATH:/path/to/isaac_scripts`.
2. Navigate to the `OmniIsaacGymEnvs/omniisaacgymenvs` folder.
3. Run `/path/to/your/isaac-sim/python.sh scripts/rl_train.py task=KinovaTask`.
4. Additional arguments you can pass to that script include:
    - `headless=True`
    - `num_envs=<how many robots you want to spawn>`
    - `test=True` if you want to examine a policy
    - `checkpoint=/path/to/a/checkpoint` (Note that you always have to do this to examine a policy. Just setting `test` to `True` will not load a trained policy)
    - `max_iterations=<how many epochs to run>` The default is 100 and is pretty quick (about 400,000 timesteps with 256 robots). Setting to 1000 is pretty long but gives very good results.
