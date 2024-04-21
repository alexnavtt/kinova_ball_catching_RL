import math
from scipy.spatial.transform import Rotation

import numpy as np
from scipy.spatial.transform import Rotation
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.prims import create_prim
from omni.isaac.core.utils.stage import add_reference_to_stage, get_current_stage
from omni.isaac.core.utils.viewports import set_camera_view
from omni.isaac.core.objects import DynamicSphere
from omni.isaac.core.prims.rigid_prim_view import RigidPrimView

from omniisaacgymenvs.tasks.base.rl_task import RLTask

import os
import torch
from gym import spaces
from kinova import Kinova
from omni.isaac.core.utils.prims import get_prim_at_path
from kinova_view import KinovaView
from omni.isaac.dynamic_control import _dynamic_control
from omni.isaac.core.utils.extensions import enable_extension
enable_extension("omni.replicator.isaac")
enable_extension("omni.kit.window.viewport")
import omni.replicator.core as rep

dc = _dynamic_control.acquire_dynamic_control_interface()

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
            "min_dist"  : 2.0,
            "collisions": 10.0,
            "rel_vel"   : 2.0,
            "alignment" : 1.0,
            "catch"     : 50.0
        }

        # Set the action space
        self.action_space = spaces.Box(
            np.concatenate((self._robot_lower_joint_limits, self._gripper_lower_joint_limit)),
            np.concatenate((self._robot_upper_joint_limits, self._gripper_upper_joint_limit))
        )

        # Set the observation space (robot joints plus the ball xyz)
        self.observation_space = spaces.Box(
            np.concatenate((self._robot_lower_joint_limits, self._gripper_lower_joint_limit, -1.0*np.ones(6,dtype=np.float32)*np.Inf)),
            np.concatenate((self._robot_upper_joint_limits, self._gripper_upper_joint_limit, +1.0*np.ones(6,dtype=np.float32)*np.Inf))
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

            # register the ArticulationView object to the world, so that it can be initialized
            scene.add(self._kinovas)
            scene.add(self._lfingers)
            scene.add(self._rfingers)
            scene.add(self._balls)

            # set default camera viewport position and target
            self.set_initial_camera_params()

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

        # TODO: Try removing this line in set_up_scene and see if it works
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
        scene.add(self._kinovas)
        scene.add(self._balls)
        scene.add(self._lfingers)
        scene.add(self._rfingers)

    def get_kinova(self):
        self._kinova: Kinova = Kinova(
            prim_path=f"{self.default_zero_env_path}/kinova", 
            name="kinova"
        )

        # Add an invisible depth camera to the robot's wrist (Maybe later)
        # rp_wrist = rep.create.render_product(f"{prim_path}/end_effector_link/Camera_Xform/WristCam", (102, 51))
        # self._depth_camera = rep.AnnotatorRegistry.get_annotator("distance_to_camera")
        # self._depth_camera.attach(rp_wrist)

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
        ball_vel = self._balls.get_velocities(clone=False)[:,:3]
        ball_pos, _ = self._balls.get_world_poses(clone=False)
        ball_pos = ball_pos - self._env_pos

        # Get the end effector position
        lfinger_pos, _ = self._lfingers.get_world_poses()
        rfinger_pos, _ = self._rfingers.get_world_poses()
        end_effector_pos = (lfinger_pos + rfinger_pos)/2 - self._env_pos

        # Extract the information that we need for the model
        joint_pos   = dof_pos[:, 0:7]
        gripper_pos = dof_pos[:, self._gripper_dof_index_1].unsqueeze(1)

        # Record the data that we want to use in the reward and is_done functions
        self.ball_dist  = torch.norm(ball_pos - end_effector_pos, dim=1)
        self.ball_speed = torch.norm(ball_vel, dim=1)
        self.ball_height = ball_pos[:, 2]

        self.obs_buf = torch.cat(
            (
                joint_pos,
                gripper_pos,
                ball_pos,
                ball_vel
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
        # full_dof_actions = self._kinovas.get_joint_positions(indices=[0])[0]
        # joint_state_actions[-1] = math.radians(46)
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

    # def calculate_metrics(self) -> float:
    #     # use states from the observation buffer to compute reward
    #     # TODO: Alter this for multiple robots
    #     joint_angles = self.obs[:7]
    #     gripper_angle = self.obs[7]
    #     ball_pos = np.array(self.obs[8:11])
    #     ball_vel = np.array(self.obs[11:])

    #     left_finger_pose = dc.get_rigid_body_pose(dc.get_rigid_body(self._kinova.prim_path + "/robotiq_85_left_finger_tip_link"))
    #     right_finger_pose = dc.get_rigid_body_pose(dc.get_rigid_body(self._kinova.prim_path + "/robotiq_85_right_finger_tip_link"))
    #     end_effector_pose = dc.get_rigid_body_pose(dc.get_rigid_body(self._kinova.prim_path + "/end_effector_link"))
    #     end_effector_z_axis = Rotation.from_quat(end_effector_pose.r).as_matrix()[:3,2]
    #     ball_vel_axis = ball_vel / np.linalg.norm(ball_vel)
    #     alignment = -1.0*np.dot(end_effector_z_axis, ball_vel_axis)

    #     gripper_pos = 0.5*(np.array(left_finger_pose.p) + np.array(right_finger_pose.p))

    #     gripper_vel = dc.get_rigid_body_linear_velocity(dc.get_rigid_body(self._kinova.prim_path + "/robotiq_85_base_link"))
    #     relative_vel = np.array(gripper_vel) - ball_vel

    #     # Success - whether the ball collided with the gripper
    #     # d_min   - Minimum distance between the ball and the gripper
    #     # H       - Whether we maintained hold of the ball
    #     # C       - Whether we collided with robot or the ground
    #     # reward = lambda1 * success - lambda2 * d_min + lambda3 * H - lambda4 * C
    #     ball_gripper_dist = np.linalg.norm(gripper_pos - ball_pos)

    #     reward = 0
    #     # reward += self._weights["min_dist"  ]*ball_gripper_dist*-1.0
    #     reward += self._weights["min_dist"  ]*(1.0/(ball_gripper_dist + 1.0))
    #     reward *= self._weights["alignment" ]*alignment
    #     # reward += self._weights["rel_vel"   ]*(np.linalg.norm(relative_vel))*-1.0
    #     reward += self._weights["collisions"]*(ball_gripper_dist < 0.10)
    #     reward += self._weights["catch" ]*(np.linalg.norm(ball_vel)<0.1 and ball_pos[2]>0.1)
    #     reward += 1/(1 + gripper_angle)

    #     self._reward_over_time.append(reward)
    #     return reward

    def calculate_metrics(self) -> dict:
        ball_dist_reward = 1.0/(1.0 + self.ball_dist)
        self.rew_buf[:] = ball_dist_reward

    def is_done(self):
        self.reset_buf = torch.where(self.ball_height < 0.1, torch.ones_like(self.reset_buf), self.reset_buf)

    # def is_done(self) -> None:
    #     # cart_pos = self.obs[:, 0]
    #     # pole_pos = self.obs[:, 2]
    #     ball_pos = self.obs[8:11].clone().detach()  # Using clone().detach() as recommended
    #     ball_vel = self.obs[11:].clone().detach()  # Same here
    #     # ball_pos = torch.tensor(self.obs[8:11])  # ensure get_observations has ball_pos set in this column
    #     # ball_vel = torch.tensor(self.obs[11:])
    #     left_finger_pose = dc.get_rigid_body_pose(dc.get_rigid_body(self._kinova.prim_path + "/robotiq_85_left_finger_tip_link"))
    #     right_finger_pose = dc.get_rigid_body_pose(dc.get_rigid_body(self._kinova.prim_path + "/robotiq_85_right_finger_tip_link"))
    #     # gripper_pos = torch.tensor(0.5*(np.array(left_finger_pose.p) + np.array(right_finger_pose.p)))
    #     gripper_pos = torch.tensor(0.5 * (np.array(left_finger_pose.p) + np.array(right_finger_pose.p)), dtype=torch.float64)
    #     ball_gripper_dist = torch.norm(gripper_pos - ball_pos)



    #     # print(f"is_done | ball_pos: {ball_pos} | height: {ball_pos[2]} | ball_vel_norm: {torch.norm(ball_vel)}")

    #     # TODO: Handle if the robot catches the ball... Done

    #     # If centroid of ball is below 10 cm, end episode
    #     resets = torch.where(ball_pos[2] < 0.1, torch.tensor(1), torch.tensor(0))

    #     # reset the robot if the ball is within a 10 cm of the grippers centroid and near stationary
    #     # combined_condition = (ball_gripper_dist < 0.10) & (torch.norm(ball_vel) < 0.01)
    #     # resets = torch.where(combined_condition, torch.tensor(1), resets)

    #     # print(f"Resets: {resets}")
    #     self.resets = resets

        # return resets.item() or self._num_frames > self._max_episode_length
