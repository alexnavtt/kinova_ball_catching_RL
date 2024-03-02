import omni.kit.commands
from omni.importer.urdf import _urdf

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
    urdf_path="/home/alex/workspaces/cs395T/src/kinova_ball_catching_RL/models/kinova.urdf", 
    import_config=import_config,
    dest_path="/home/alex/workspaces/cs395T/src/kinova_ball_catching_RL/models/kinova.usd"
)
print(f"Model loaded: {prim_path}, {result}")
