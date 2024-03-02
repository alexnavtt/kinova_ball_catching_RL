import asyncio
import numpy as np

from omni.isaac.core import World
from omni.isaac.core.utils.viewports import set_camera_view
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.utils.stage import clear_stage, add_reference_to_stage 

async def init():

    # Clear anything already in the simulation
    clear_stage()

    # Set up the simulation world
    if World.instance():
        World.instance().clear_instance()
    world: World = World()
    await world.initialize_simulation_context_async()
    world.scene.add_default_ground_plane()
    set_camera_view(np.array([0, 0.3, 1.3]), np.array([0, 0, 1.3]))

    # Load the robot TODO: Figure out relative paths
    prim_path = "/World/kinova"
    print("Loading Kinova into stage")
    add_reference_to_stage(usd_path="/home/alex/workspaces/cs395T/src/kinova_ball_catching_RL/models/kinova_closed_loop.usd", prim_path=prim_path)
    print("Loaded")

    # Create a manipulator object out of the model
    print("Creating manipulator")
    try:
        my_kinova: Articulation = world.scene.add(
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
    print(f"Kinova has {my_kinova.num_dof} DoFs")
    my_kinova.apply_action(ArticulationAction(joint_positions=np.array([0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0])*0.3))
    print("Done")

asyncio.ensure_future(init())
