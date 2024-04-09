

from omni.isaac.gym.vec_env import VecEnvBase

# Create the environment
env = VecEnvBase(headless=False)  # Set headless=True if you don't need a GUI

# Initialize and set your task
from kinova_task import KinovaTask  # Import your task class here
task = KinovaTask(name="KinovaCatchBall")
env.set_task(task, backend="torch")

# A simple loop to keep the window open, you may replace it with a more complex test
env.reset()
while True:
    env.step([0])  # Passing a dummy action, replace with actual action space

#--------------------------------------------------------------------------

# # Initialize the environment
# from omni.isaac.gym.vec_env import VecEnvBase
# import numpy as np  
# env = VecEnvBase(headless=False)

# # Create and set up your custom task
# from kinova_task import KinovaTask  
# task = KinovaTask(name="KinovaCatchBall")
# env.set_task(task)

# # Main loop to keep the simulation running
# try:
#     print("Simulation running... Press CTRL+C to exit.")
#     while True:
#         # Generate a dummy action for each step. Adjust the size based on your action space definition.
#         dummy_action = np.random.uniform(low=-1.0, high=1.0, size=(4,))
        
#         # Pass the dummy action to the env.step() method
#         observation, reward, done, info = env.step(dummy_action)  # Adjust as necessary based on your environment's API
# except Exception as e:
#     print(f"Unexpected error: {e}")
# finally:
#     env.close()  # Ensure the environment is closed properly