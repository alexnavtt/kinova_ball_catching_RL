#launch Isaac Sim before any other imports
#default first two lines in any standalone application
from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": False}) # we can also run as headless.

from omni.isaac.core import World
from omni.isaac.core.objects import DynamicSphere
import numpy as np

world = World()
world.scene.add_default_ground_plane()
tennis_ball =  world.scene.add(
 DynamicSphere(
                prim_path="/World/random_cube",
                name="tennis_ball",
                position=np.array([0.3, 0.3, 0.3]),
                scale=np.array([0.0515, 0.0515, 0.0515]),
                color=np.array([0, 1.0, 0]),
                linear_velocity = np.array([1, -1, 0])
            ))
# Resetting the world needs to be called before querying anything related to an articulation specifically.
# Its recommended to always do a reset after adding your assets, for physics handles to be propagated properly
world.reset()
for i in range(500):
    position, orientation = tennis_ball.get_world_pose()
    linear_velocity = tennis_ball.get_linear_velocity()
    # will be shown on terminal
    print("Tennis ball position is : " + str(position))
    print("Tennis ball orientation is : " + str(orientation))
    print("Tennis ball linear velocity is : " + str(linear_velocity))
    # we have control over stepping physics and rendering in this workflow
    # things run in sync
    world.step(render=True) # execute one physics step and one rendering step

simulation_app.close() # close Isaac Sim

