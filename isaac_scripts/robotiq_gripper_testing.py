import os
import math
import time
import asyncio
import numpy as np

from omni.isaac.core import World
from omni.isaac.core.objects import DynamicSphere
from omni.isaac.core.utils.viewports import set_camera_view
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.utils.stage import clear_stage, add_reference_to_stage 
import omni.replicator.core as rep
import omni.syntheticdata as sd

my_kinova: Articulation

async def move():
    print("Resetting to zero")
    print(f"My kinova: {my_kinova}")
    my_kinova.apply_action(ArticulationAction(joint_positions=np.array([0, 2, 0, 6, 5, -5, 3, 0, 0, 0, 0, 0, 0])*0.3))
    # my_kinova.apply_action(ArticulationAction(joint_positions=np.array([0, 0, 0 ,0 ,0 ,0 ,0, math.radians(40), 0, 0, 0, 0, 0])))

async def test_bbox_3d():

    # distance_light = rep.create.light(rotation=(315,0,0), intensity=3000, light_type="distant")

    cone = rep.create.cone(semantics=[("prim", "cone")], position=(1, 0, 0))
    sphere = rep.create.sphere(semantics=[("prim", "sphere")], position=(-1, 0, 0))
    invalid_type = rep.create.cube(semantics=[("shape", "boxy")], position=(0, 1, 0))

    # Setup semantic filter
    sd.SyntheticData.Get().set_instance_mapping_semantic_filter("prim:*")

    cam = rep.create.camera(position=(5,5,5), look_at=cone)
    rp = rep.create.render_product(cam, (1024, 512))

    bbox_3d = rep.AnnotatorRegistry.get_annotator("bounding_box_3d")
    bbox_3d.attach(rp)

    data = bbox_3d.get_data()
    print(data)

async def init():
    pass
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
    print(f"Current directory: {__file__}")
    add_reference_to_stage(usd_path="/home/alex/workspaces/cs395T/src/kinova_ball_catching_RL/models/kinova_closed_loop.usd", prim_path=prim_path)
    # add_reference_to_stage(usd_path="../models/kinova_closed_loop.usd", prim_path=prim_path)
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
                prim_path=prim_path, 
                name="kinova_robot",
            )
        )
        print(f"My kinova: {my_kinova}")
        await world.reset_async()

    except Exception as e:
        print(f"Error: {e}")

    # Create depth image object
    wrist_cam = rep.create.camera(
        position = (0.0, -0.05, 0.0), 
        look_at=(0.0, 0.0, 10.0), 
        parent="/World/kinova/robotiq_85_base_link",
        clipping_range=(0.10, 1000000.0)
    )
    rp_wrist = rep.create.render_product(wrist_cam, (5, 5))
    depth_camera = rep.AnnotatorRegistry.get_annotator("distance_to_camera")
    depth_camera.attach(rp_wrist)

    # Use the manipulator
    print(f"Kinova has {my_kinova.num_dof} DoFs")
    my_kinova.apply_action(ArticulationAction(joint_positions=np.array([0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0])*0.5))
    print("Done")

    await move()
    print("Finished move")

    await rep.orchestrator.step_async()
    data = depth_camera.get_data()
    print(data)

asyncio.ensure_future(init())
#asyncio.ensure_future(test_bbox_3d())



