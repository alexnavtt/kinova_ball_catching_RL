import time
import asyncio
import numpy as np

from omni.isaac.core import World
from omni.isaac.core.objects import DynamicSphere
from omni.isaac.core.utils.viewports import set_camera_view
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.utils.stage import clear_stage, add_reference_to_stage 

my_kinova: Articulation

async def move():
    # await asyncio.sleep(2)
    print("Resetting to zero")
    print(f"My kinova: {my_kinova}")
    my_kinova.apply_action(ArticulationAction(joint_positions=np.array([0, 2, 0, 6, 5, -5, 3, 0, 0, 0, 0, 0, 0])*0.3))

async def init():
    global my_kinova

    # Clear anything already in the simulation
    clear_stage()

    # Set up the simulation world
    if World.instance():
        World.instance().clear_instance()
    world: World = World()
    await world.initialize_simulation_context_async()
    world.scene.add_default_ground_plane()
    set_camera_view(np.array([-1.5, 2.0, 1.5]), np.array([0, 0, 0.4]))

    # Load the robot TODO: Figure out relative paths
    prim_path = "/World/kinova"
    print("Loading Kinova into stage")
    add_reference_to_stage(usd_path="/home/alex/workspaces/cs395T/src/kinova_ball_catching_RL/models/kinova_closed_loop.usd", prim_path=prim_path)
    await world.reset_async()
    print("Loaded")

    # Load in the sphere to the simulation
    DynamicSphere(
        prim_path="/World/sphere",
        position=np.array([0.3, 0.2, 1.0]),
        scale=np.array([.03, .03, .03]),
        color=np.array([.2,.3,0.])
    )

    # Create a manipulator object out of the model
    print("Creating manipulator")
    try:
        my_kinova = world.scene.add(
            Articulation(
                prim_path=prim_path + "/world", 
                name="kinova_robot",
            )
        )
        print(f"My kinova: {my_kinova}")
        await world.reset_async()

    except Exception as e:
        print(f"Error: {e}")

    # Use the manipulator
    # print(f"Kinova has {my_kinova.num_dof} DoFs")
    # my_kinova.apply_action(ArticulationAction(joint_positions=np.array([0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0])*0.5))
    # print("Done")

    await move()

asyncio.ensure_future(init())


