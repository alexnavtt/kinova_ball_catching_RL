import asyncio
import numpy as np
import omni.kit.commands

from omni.isaac.core import World, PhysicsContext
from omni.isaac.core.utils.viewports import set_camera_view
from omni.importer.urdf import _urdf
from omni.isaac.manipulators import SingleManipulator
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.utils.stage import clear_stage, add_reference_to_stage

def load_urdf():

    urdf_interface = _urdf.acquire_urdf_interface()
    import_config = _urdf.ImportConfig()
    import_config.merge_fixed_joints = False
    import_config.convex_decomp = False
    import_config.fix_base = True
    import_config.make_default_prim = True
    import_config.self_collision = False
    import_config.create_physics_scene = False
    import_config.import_inertia_tensor = True
    import_config.default_drive_strength = 1000.0
    import_config.default_position_drive_damping = 100.0
    import_config.default_drive_type = _urdf.UrdfJointTargetType.JOINT_DRIVE_POSITION
    import_config.distance_scale = 1
    import_config.density = 0.0
    import_config.set_parse_mimic(False)

    print("Loading Kinova model")
    # TODO: Figure out relative paths
    result, prim_path = omni.kit.commands.execute(
        "URDFParseAndImportFile", 
        urdf_path="/home/alex/kinova.urdf", 
        import_config=import_config,
        dest_path="/home/alex/workspaces/cs395T/src/kinova_ball_catching_RL/models/kinova.usd"
    )
    print(f"Model loaded: {prim_path}, {result}")    

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

    # load_urdf()

    # Load the robot TODO: Figure out relative paths
    prim_path = "/World/kinova"
    print("Loading Kinova into stage")
    add_reference_to_stage(usd_path="/home/alex/workspaces/cs395T/src/kinova_ball_catching_RL/models/kinova.usd", prim_path=prim_path)
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
    my_kinova.set_joint_positions(np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, -1, -1, -1, 1])*0.6)
    print("Done")

asyncio.ensure_future(init())
