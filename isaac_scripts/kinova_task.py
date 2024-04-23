import math
import torch
import numpy as np

from gym import spaces
from scipy.spatial.transform import Rotation
from omni.isaac.core.objects import DynamicSphere
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.utils.stage import get_current_stage
from omni.isaac.core.utils.viewports import set_camera_view
from omni.isaac.core.prims.rigid_prim_view import RigidPrimView
from omni.isaac.core.utils.extensions import enable_extension
enable_extension("omni.replicator.isaac")
from omni.replicator.core.scripts.annotators import Annotator

from kinova import Kinova
from omniisaacgymenvs.tasks.base.rl_task import RLTask

class KinovaTask(RLTask):
    def __init__(
            self, 
            name, 
            sim_config,
            env,
            offset=None
    ) -> None:
        
        # Simulation and training parameters
        self.update_config(sim_config)
        self.dt = 1/60
        self._num_observations = 14     # 7 robot joint states, 1 gripper joint state and 6 ball states
        self._num_actions = 8           # 7 arm joint actions and 1 gripper joint action

        # Task-specific parameters (fill in as you need them)
        self._robot_lower_joint_limits  = np.deg2rad([-180.0, -128.9, -180.0, -147.8, -180.0, -120.3, -180.0], dtype=np.float32)
        self._robot_upper_joint_limits  = -1.0 * self._robot_lower_joint_limits
        self._gripper_lower_joint_limit = np.deg2rad([ 0.0], dtype=np.float32)
        self._gripper_upper_joint_limit = np.deg2rad([46.0], dtype=np.float32)

        # Set the reward function weights
        self._weights = {
            "min_dist"  : 1.0,
            "collisions": 1.0,
            "rel_vel"   : 1.0,
            "alignment" : 1.0,
            "catch"     : 1.0
        }

        # Define the observation and action spaces of the underlying neural network
        robot_state_space = {
            "low" : np.concatenate((self._robot_lower_joint_limits, self._gripper_lower_joint_limit)),
            "high": np.concatenate((self._robot_upper_joint_limits, self._gripper_upper_joint_limit)),
        }

        ball_state_space = {
            "low" : -1.0*np.ones(6,dtype=np.float32)*np.Inf,
            "high": +1.0*np.ones(6,dtype=np.float32)*np.Inf,
        }

        self.observation_space = spaces.Box(
            low =np.concatenate((robot_state_space["low" ], ball_state_space["low"])),
            high=np.concatenate((robot_state_space["high"], ball_state_space["high"]))
        )

        self.action_space = spaces.Box(
            low = robot_state_space["low"],
            high= robot_state_space["high"]
        )

        # Record the reward over time for plotting 
        self._reward_over_time = []

        # trigger __init__ of parent class
        RLTask.__init__(self, name, env)

    def update_config(self, sim_config):
        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config

        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]
        self._max_episode_length = self._task_cfg["env"]["episodeLength"]

    def set_up_scene(self, scene) -> None:
            self.stage = get_current_stage()

            # first create a single environment
            self.get_kinova()
            self.get_ball(scene)

            # call the parent class to clone the single environment
            super().set_up_scene(scene, replicate_physics=False)

            # construct an ArticulationView object to hold our collection of environments
            self._kinovas = ArticulationView(
                prim_paths_expr=f"{self.default_base_env_path}/.*/kinova", 
                name="kinova_view",
                reset_xform_properties=False
            )

            self._balls = RigidPrimView(
                prim_paths_expr=f"{self.default_base_env_path}/.*/ball", 
                name="ball_view",
                reset_xform_properties=False
            )

            self._lfingers = RigidPrimView(
                prim_paths_expr=f"{self.default_base_env_path}/.*/kinova/robotiq_85_left_finger_tip_link",
                name = "left_fingers_view",
                reset_xform_properties=False
            )

            self._rfingers = RigidPrimView(
                prim_paths_expr=f"{self.default_base_env_path}/.*/kinova/robotiq_85_right_finger_tip_link",
                name = "right_fingers_view",
                reset_xform_properties=False
            )
            self._hands = RigidPrimView(
                prim_paths_expr=f"{self.default_base_env_path}/.*/kinova/robotiq_85_base_link",
                name = "hands_view",
                reset_xform_properties=False
            )

            # register the ArticulationView objects to the world, so that they can be initialized
            scene.add(self._kinovas)
            scene.add(self._lfingers)
            scene.add(self._rfingers)
            scene.add(self._balls)
            scene.add(self._hands)

            # set default camera viewport position and target
            self.set_initial_camera_params()

            # Add an invisible depth camera to each robot's wrist
            if self._task_cfg["sim"]["enable_cameras"]:
                self.rep.orchestrator._orchestrator._is_started = True
                self.depth_cams: list[Annotator] = []
                for robot_path in self._kinovas.prim_paths:
                    camera_path = f"{robot_path}/end_effector_link/Camera_Xform/WristCam"
                    rp_wrist = self.rep.create.render_product(camera_path, resolution=(10, 10))
                    depth_camera = self.rep.AnnotatorRegistry.get_annotator("distance_to_camera")
                    depth_camera.attach(rp_wrist)
                    self.depth_cams.append(depth_camera)

            return
    
    def initialize_views(self, scene):
        super().initialize_views(scene)
        if scene.object_exists("kinova_view"):
            scene.remove_object("kinova_view", registry_only=True)
        if scene.object_exists("ball_view"):
            scene.remove_object("ball_view", registry_only=True)
        if scene.object_exists("left_fingers_view"):
            scene.remove_object("left_fingers_view", registry_only=True)
        if scene.object_exists("right_fingers_view"):
            scene.remove_object("right_fingers_view", registry_only=True)
        if scene.object_exists("hands_view"):
            scene.remove_object("hands_view", registry_only=True)

        self._kinovas = ArticulationView(
            prim_paths_expr=f"{self.default_zero_env_path}/.*/kinova", 
            name="kinova_view",
            reset_xform_properties=False
        )
        self._balls = RigidPrimView(
            prim_paths_expr=f"{self.default_base_env_path}/.*/ball", 
            name="ball_view",
            reset_xform_properties=False
        )
        self._lfingers = RigidPrimView(
            prim_paths_expr=f"{self.default_base_env_path}/.*/kinova/robotiq_85_left_finger_tip_link",
            name = "left_fingers_view",
            reset_xform_properties=False
        )
        self._rfingers = RigidPrimView(
            prim_paths_expr=f"{self.default_base_env_path}/.*/kinova/robotiq_85_right_finger_tip_link",
            name = "right_fingers_view",
            reset_xform_properties=False
        )
        self._hands = RigidPrimView(
            prim_paths_expr=f"{self.default_base_env_path}/.*/kinova/robotiq_85_base_link",
            name = "hands_view",
            reset_xform_properties=False
        )
        scene.add(self._kinovas)
        scene.add(self._balls)
        scene.add(self._lfingers)
        scene.add(self._rfingers)
        scene.add(self._hands)

    def get_kinova(self):
        self._kinova_base: Kinova = Kinova(
            prim_path=f"{self.default_zero_env_path}/kinova", 
            name="kinova"
        )

    def get_ball(self, scene):
        # Add and launch the ball (this should maybe be moved to post_reset?)
        ball_radius = 0.03
        self._ball = scene.add(
            DynamicSphere(
                prim_path=f"{self.default_zero_env_path}/ball",
                name="tennis_ball",
                position=np.array([0.3, 0.3, 0.3]),
                scale=np.ones(3)*ball_radius,
                color=np.array([0, 1.0, 0]),
            )
        )

    def set_initial_camera_params(self, camera_position=None, camera_target=None):
        camera_position = camera_position if camera_position is not None else [5, -7, 3] 
        camera_target = camera_target if camera_target is not None else [0, 0, 1]
        set_camera_view(
            eye=camera_position,
            target=camera_target,
            camera_prim_path="/OmniverseKit_Persp",
        ) 

    def get_observations(self):
        # Get the robot and ball state from the simulation
        dof_pos  = self._kinovas.get_joint_positions(clone=False)
        ball_vel = self._balls.get_velocities(clone=False)
        ball_pos, _ = self._balls.get_world_poses(clone=False)
        ball_pos = ball_pos - self._env_pos

        # Get the end effector position
        lfinger_pos, _ = self._lfingers.get_world_poses(clone=False)
        rfinger_pos, _ = self._rfingers.get_world_poses(clone=False)
        end_effector_pos = (lfinger_pos + rfinger_pos)/2 - self._env_pos

        # Get the vector from the gripper to the ball
        self.ball_offset        = ball_pos - end_effector_pos
        self.ball_gripper_dist  = torch.norm(self.ball_offset, p=2, dim=-1)

        # Get gripper and orientation (must be done on CPU because of the scipy function call)
        hand_quats = self._hands.get_world_poses()[1].cpu()
        hand_orientations = torch.zeros((self._num_envs, 3), device="cpu")
        for i in range(self._num_envs):
            gripper_quat = hand_quats[i]
            z_axis = Rotation.from_quat([gripper_quat[1], gripper_quat[2], gripper_quat[3], gripper_quat[0]]).as_matrix()[:3,2]
            hand_orientations[i] = torch.tensor(z_axis, device="cpu", dtype=torch.float32)
        self.hand_orientations = hand_orientations.to(device=self._device, dtype=torch.float32)

        # Record the gripper linear velocity
        self.gripper_vel = self._hands.get_velocities()[:, :3]

        # Extract the information that we need for the model
        joint_pos   = dof_pos[:, 0:7]
        gripper_pos = dof_pos[:, self._gripper_dof_index_1].unsqueeze(1)

        # Record the data that we want to use in the reward and is_done functions
        self.ball_vel      = ball_vel[:,:3]
        self.ball_speed    = torch.norm(ball_vel, p=2, dim=-1)
        self.ball_vel_axis = self.ball_vel/self.ball_speed.unsqueeze(dim=1)
        self.ball_height   = ball_pos[:, 2]

        self.obs_buf = torch.cat(
            (
                joint_pos,
                gripper_pos,
                self.ball_offset, # Take in offset to ball instead of absolute ball position
                ball_vel[:, :3]
            ),
            dim=-1
        )
        
        observations = {
            self._kinovas.name: {
                "obs_buf": self.obs_buf
            }
        }
        return observations
    
    def pre_physics_step(self, actions: torch.Tensor) -> None:
        """
        This method will be called from VecEnvBase before each simulation step, and will pass in actions from the RL policy as an argument
        """
        if not self.world.is_playing():
            return

        # Check to see if any environments need resetting
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)

        joint_state_actions = actions.clone().to(self._device)

        full_dof_actions = torch.zeros((self._num_envs, self._kinovas.num_dof), dtype=torch.float32)
        full_dof_actions[:,:7] = joint_state_actions[:,:7]
        full_dof_actions[:, self._gripper_dof_index_1] = joint_state_actions[:, -1]
        full_dof_actions[:, self._gripper_dof_index_2] = -1.0 * joint_state_actions[:, -1]

        indices = torch.arange(self._kinovas.count, dtype=torch.int32, device=self._device)
        self._kinovas.set_joint_position_targets(full_dof_actions, indices=indices)

    def reset_idx(self, env_ids: torch.Tensor):
        # Set up reset bookkeeping
        indicies = env_ids.to(dtype=torch.int32)
        num_indices = len(indicies)

        default_pos = torch.tensor([0, 0.6, 0, 1.8, 1.5, -1.5, -0.9, 0, 0, 0, 0, 0, 0], device=self._device)
        dof_pos = torch.zeros((num_indices, self._kinovas.num_dof), device=self._device)
        dof_vel = torch.zeros((num_indices, self._kinovas.num_dof), device=self._device)
        dof_pos[:, :] = default_pos

        # Reset the robots to their default positions and zero velocity
        self._kinovas.set_joint_positions(dof_pos, indices=indicies)
        self._kinovas.set_joint_position_targets(dof_pos, indices=indicies)
        self._kinovas.set_joint_velocities(dof_vel, indices=indicies)

        # Set the balls to their default positions
        ball_positions = torch.tensor([0.5, -2.0, 0.3], device=self._device).repeat((num_indices, 1)) + self._env_pos.index_select(index=indicies, dim=0)
        self._balls.set_world_poses(positions=ball_positions, indices=indicies)

        # Sample random velocities for each reset ball
        ball_vels = torch.zeros((num_indices, 6), device=self.device)
        for i in range(num_indices):
            ball_vels[i, :3] = torch.tensor(
                self.sample_launch_velocity(
                    speed=5, 
                    cone_axis=[0,0.8660254037844386,0.5], 
                    cone_angle=15
                )
            )
        self._balls.set_velocities(ball_vels, indices=indicies)

        # More bookkeeping
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

    def post_reset(self):
        """
        This gets executed once the scene is constructed and simulation starts running
        """
        self._gripper_dof_index_1 = self._kinovas.get_dof_index("robotiq_85_left_knuckle_joint")
        self._gripper_dof_index_2 = self._kinovas.get_dof_index("robotiq_85_right_knuckle_joint")

        # Reset all environments
        indices = torch.arange(self._num_envs, dtype=torch.int64, device=self._device)
        self.reset_idx(indices)

    def sample_launch_velocity(self, speed, cone_axis, cone_angle) -> list:
        """
        Samples a random launch velocity within a velocity cone defined by a cone axis, cone angle, and speed

        Args:
        speed: The magnitude of the velocity to be sampled
        cone_axis: A list representing the direction vector of the cone axis
        cone_angle: The angle in degrees that bounds the velocity cone. 0 deg corresponds to straight line.

        Returns: A list representing the sampled velocity [vx, vy, vz]
        """

        # Normalize cone axis
        U = np.array(cone_axis) / np.linalg.norm(cone_axis)

        # # Convert cone angle to radians
        b = np.radians(cone_angle)

        # Randomly rotate about the cone axis
        theta = np.random.uniform(-b,b)
        axis_ang_rot_operator_x = Rotation.from_rotvec(theta*np.array([1,0,0]))
        X = axis_ang_rot_operator_x.apply(U)

        # Randomly rotate about x-axis axis
        # Randomly rotate about the cone axis
        phi = np.random.uniform(0,2*math.pi)
        axis_ang_rot_operator = Rotation.from_rotvec(phi*U)
        X = axis_ang_rot_operator.apply(X)

        # Scale by given speed
        X = speed*X

        return X
    
    def calculate_metrics(self) -> None:
        """
        Calculates a reward metric for the learning agent based five factors:
            - Distance from the gripper to the ball
            - Alignment of the gripper with the axis of motion of the ball
            - Relative velocity between the gripper and the ball
            - Whether the ball is within a certain threshold distance of the gripper
            - Whether the ball is stationary 
        """
        reward = torch.zeros_like(self.rew_buf)

        # Distance reward
        reward += self._weights["min_dist"] * 1.0/(1.0 + self.ball_gripper_dist)

        # Alignment reward
        reward += self._weights["alignment"] * torch.sum(self.hand_orientations * self.ball_vel_axis, dim=-1) * -1.0

        # Relative velocity reward
        relative_speed = torch.norm(self.ball_vel - self.gripper_vel, p=2, dim=-1)
        reward += self._weights["rel_vel"] * 1.0/(1.0 + relative_speed)

        # Ball in range of hand reward (i.e. "collision")
        reward += self._weights["collisions"] * (self.ball_gripper_dist < 0.1).to(dtype=torch.float32)

        # Catching reward
        reward += self._weights["catch"] * (self.ball_speed < 0.1).to(dtype=torch.float32)

        self.rew_buf[:] = reward

    def is_done(self):
        self.reset_buf = torch.where(self.ball_height < 0.1, torch.ones_like(self.reset_buf), self.reset_buf)
        self.reset_buf = torch.where(self.progress_buf >= self._max_episode_length-1, torch.ones_like(self.reset_buf), self.reset_buf)

