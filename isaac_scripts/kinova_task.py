import math
from scipy.spatial.transform import Rotation

import numpy as np
from scipy.spatial.transform import Rotation
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.tasks.base_task import BaseTask
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.prims import create_prim
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.viewports import set_camera_view
from omni.isaac.core.objects import DynamicSphere
import os
import torch
from gymnasium import spaces
from kinova import Kinova
from omni.isaac.core.utils.prims import get_prim_at_path
from kinova_view import KinovaView
from omni.isaac.dynamic_control import _dynamic_control
from omni.isaac.core.utils.extensions import enable_extension
enable_extension("omni.replicator.isaac")
enable_extension("omni.kit.window.viewport")
import omni.replicator.core as rep

dc = _dynamic_control.acquire_dynamic_control_interface()

class KinovaTask(BaseTask):
    def __init__(self, name, offset=None):
        # self.update_config(sim_config)
        self._max_episode_length = 200

        # Task-specific parameters (fill in as you need them)
        self._robot_lower_joint_limits  = np.deg2rad([-180.0, -128.9, -180.0, -147.8, -180.0, -120.3, -180.0], dtype=np.float32)
        self._robot_upper_joint_limits  = -1.0 * self._robot_lower_joint_limits
        self._gripper_lower_joint_limit = np.deg2rad([ 0.0], dtype=np.float32)
        self._gripper_upper_joint_limit = np.deg2rad([46.0], dtype=np.float32)

        # Set the reward function weights
        self._weights = {
            "min_dist"  : 2.0,
            "collisions": 3.0,
            "rel_vel"   : 2.0,
            "alignment" : 3.0,
            "catch" : 5.0
        }

        # Values used for defining RL buffers (TODO: Image input)
        self._num_observations = 14 # 7 robot joint states, 1 gripper joint state and 6 ball states
        self._num_actions = 8 # 7 arm joint actions and 1 gripper joint action
        self._device = "cpu"
        self.num_envs = 1

        # A few class buffers to store RL-related states
        self.obs = torch.zeros((self.num_envs, self._num_observations))
        self.resets = torch.zeros((self.num_envs, 1))

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

        # trigger __init__ of parent class
        BaseTask.__init__(self, name=name, offset=offset)

    def update_config(self, sim_config):
        # extract task config from main config dictionary
        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config

        # parse task config parameters
        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]
        self._kinova_positions = torch.tensor([0.0, 0.0, 2.0])

        # reset and actions related variables
        self._reset_dist = self._task_cfg["env"]["resetDist"]
        self._max_push_effort = self._task_cfg["env"]["maxEffort"]

    def set_up_scene(self, scene) -> None:
            # first create a single environment
            self.get_kinova()

            # call the parent class to clone the single environment
            super().set_up_scene(scene)

            # construct an ArticulationView object to hold our collection of environments
            self._kinovas = ArticulationView(
                prim_paths_expr="/World/envs/kinova", name="kinova_view"
            )

            # register the ArticulationView object to the world, so that it can be initialized
            scene.add_default_ground_plane()
            scene.add(self._kinovas)

            # Add and launch the ball (this should maybe be moved to post_reset?)
            ball_radius = 0.03
            self._ball = scene.add(
                DynamicSphere(
                    prim_path="/World/random_sphere",
                    name="tennis_ball",
                    position=np.array([0.3, 0.3, 0.3]),
                    scale=np.ones(3)*ball_radius,
                    color=np.array([0, 1.0, 0]),
                )
            )

            # set default camera viewport position and target
            self.set_initial_camera_params()

            return

    def get_kinova(self):
        prim_path = "/World/envs/kinova"
        self._kinova: Kinova = Kinova(prim_path=prim_path, name="kinova")

        # Add an invisible depth camera to the robot's wrist
        rp_wrist = rep.create.render_product(f"{prim_path}/end_effector_link/Camera_Xform/WristCam", (102, 51))
        self._depth_camera = rep.AnnotatorRegistry.get_annotator("distance_to_camera")
        self._depth_camera.attach(rp_wrist)

    def set_initial_camera_params(
        self, camera_position=[5, -7, 3], camera_target=[0, 0, 1]
    ):
        set_camera_view(
            eye=camera_position,
            target=camera_target,
            camera_prim_path="/OmniverseKit_Persp",
        )  # need to test this in GUI

    def post_reset(self):
        """
        This gets executed once the scene is constructed and simulation starts running
        """
        self._gripper_dof_index_1 = self._kinovas.get_dof_index("robotiq_85_left_knuckle_joint")
        self._gripper_dof_index_2 = self._kinovas.get_dof_index("robotiq_85_right_knuckle_joint")


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

    def reset(self, env_ids=None):
        """
        This method is used to set our environment into an initial state for starting a new training episode
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self._device)
        num_resets = len(env_ids)

        # zero DOF positions and velocities
        dof_pos = torch.tensor(np.array([0, 2, 0, 6, 5, -5, -3, 0, 0, 0, 0, 0, 0], dtype=np.float32)*0.3, device=self._device)
        # dof_pos = torch.zeros((num_resets, self._kinovas.num_dof), device=self._device)
        dof_vel = torch.zeros((num_resets, self._kinovas.num_dof), device=self._device)

        # reset configuration of the robot
        indices = [0]
        self._kinovas.set_joint_positions(dof_pos, indices=indices)
        self._kinovas.set_joint_velocities(dof_vel, indices=indices)

        # reset configuration of the ball
        self._ball.set_world_pose([0.5, -2.0, 0.3])
        # dc.set_rigid_body_linear_velocity(dc.get_rigid_body(self._ball.prim_path), [0.0, 5.0, 3.0])
        sampled_ball_velocity = self.sample_launch_velocity(speed=5, cone_axis=[0,0.8660254037844386,0.5], cone_angle=15)
        dc.set_rigid_body_linear_velocity(dc.get_rigid_body(self._ball.prim_path), sampled_ball_velocity)

        self._num_frames = 0
        # bookkeeping
        # self.resets[env_ids] = 0


    def pre_physics_step(self, actions) -> None:
        """
        This method will be called from VecEnvBase before each simulation step, and will pass in actions from the RL policy as an argument
        """
        # Check to see if any environments need resetting
        reset_env_ids = self.resets.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset(reset_env_ids)

        # actions = torch.tensor(actions)
        joint_state_actions = torch.tensor(actions)
        # print(f"Joint state actions: {joint_state_actions}")

        full_dof_actions = torch.zeros(13, dtype=torch.float32)
        full_dof_actions[:7] = joint_state_actions[:7]
        # full_dof_actions = self._kinovas.get_joint_positions(indices=[0])[0]
        # joint_state_actions[-1] = math.radians(46)
        full_dof_actions[self._gripper_dof_index_1] = joint_state_actions[-1]
        full_dof_actions[self._gripper_dof_index_2] = -1.0 * joint_state_actions[-1]
        # print(f"{full_dof_actions}")

        # indices = torch.arange(self._cartpoles.count, dtype=torch.int32, device=self._device)
        indices = torch.arange(
            self._kinovas.count, dtype=torch.int32, device=self._device
        )

        self._num_frames += 1

        self._kinovas.apply_action(
            ArticulationAction(
                joint_positions=full_dof_actions,
            ),
            indices=indices
        )
        # References
        # https://docs.omniverse.nvidia.com/isaacsim/latest/isaac_gym_tutorials/tutorial_gym_isaac_gym_new_oige_example.html#isaac-sim-app-tutorial-gym-omni-isaac-gym-new-example
        # https://docs.omniverse.nvidia.com/py/isaacsim/source/extensions/omni.isaac.manipulators/docs/index.html

    def get_observations(self):
        # Get the robot and ball state from the simulation
        dof_pos  = self._kinovas.get_joint_positions()
        dof_ball = self._ball.get_world_pose()
        vel_ball = dc.get_rigid_body_linear_velocity(dc.get_rigid_body(self._ball.prim_path))

        # print(f"DoF ball: {dof_ball}\nDoF pos: {dof_pos}\nVel ball: {vel_ball}")

        # Extract the information that we need for the model
        joint_pos   = dof_pos[0, 0:7]
        gripper_pos = dof_pos[0, self._gripper_dof_index_1],
        ball_pos    = dof_ball[0][0:3]
        ball_vel    = vel_ball
        #TODO include ball velocity

        # print(f"Ball pos: {ball_pos}\nJoint pos: {joint_pos}\nGripper pos: {gripper_pos}\nBall vel: {ball_vel}")

        # # populate the observations buffer                               #|
        # self.obs_buf[:, 0] = joint_pos                                   #|I added these to work with the calculate_metrics method
        # self.obs_buf[:, 1] = gripper_pos                                 #|to reflect what the tutorials do - Caleb
        # self.obs_buf[:, 2] = ball_pos                                    #|
        # # construct the observations dictionary and return               #|
        # observations = {self._cartpoles.name: {"obs_buf": self.obs_buf}} #|

        obs = np.concatenate((joint_pos, gripper_pos, ball_pos, ball_vel))
        self.obs = torch.tensor(obs)
        # print(f"Self.obs: {self.obs}")
        return self.obs

    def calculate_metrics(self) -> float:
        # use states from the observation buffer to compute reward
        # TODO: Alter this for multiple robots
        joint_angles = self.obs[:7]
        gripper_angle = self.obs[7]
        ball_pos = np.array(self.obs[8:11])
        ball_vel = np.array(self.obs[11:])

        left_finger_pose = dc.get_rigid_body_pose(dc.get_rigid_body(self._kinova.prim_path + "/robotiq_85_left_finger_tip_link"))
        right_finger_pose = dc.get_rigid_body_pose(dc.get_rigid_body(self._kinova.prim_path + "/robotiq_85_right_finger_tip_link"))
        end_effector_pose = dc.get_rigid_body_pose(dc.get_rigid_body(self._kinova.prim_path + "/end_effector_link"))
        end_effector_z_axis = Rotation.from_quat(end_effector_pose.r).as_matrix()[:3,2]
        ball_vel_axis = ball_vel / np.linalg.norm(ball_vel)
        alignment = -1.0*np.dot(end_effector_z_axis, ball_vel_axis)

        gripper_pos = 0.5*(np.array(left_finger_pose.p) + np.array(right_finger_pose.p))

        gripper_vel = dc.get_rigid_body_linear_velocity(dc.get_rigid_body(self._kinova.prim_path + "/robotiq_85_base_link"))
        relative_vel = np.array(gripper_vel) - ball_vel

        # Success - whether the ball collided with the gripper
        # d_min   - Minimum distance between the ball and the gripper
        # H       - Whether we maintained hold of the ball
        # C       - Whether we collided with robot or the ground
        # reward = lambda1 * success - lambda2 * d_min + lambda3 * H - lambda4 * C
        ball_gripper_dist = np.linalg.norm(gripper_pos - ball_pos)

        reward = 0
        # reward += self._weights["min_dist"  ]*ball_gripper_dist*-1.0
        reward += self._weights["min_dist"  ]*(1.0/(ball_gripper_dist + 1.0))
        reward += self._weights["collisions"]*(ball_gripper_dist < 0.10)
        reward += self._weights["rel_vel"   ]*(np.linalg.norm(relative_vel))*-1.0
        reward += self._weights["alignment" ]*alignment
        reward += self._weights["catch" ]*(np.linalg.norm(ball_vel)<0.1 and ball_pos[2]>0.1)
        return reward

    def is_done(self) -> None:
        # cart_pos = self.obs[:, 0]
        # pole_pos = self.obs[:, 2]
        ball_pos = self.obs[8:11].clone().detach()  # Using clone().detach() as recommended
        ball_vel = self.obs[11:].clone().detach()  # Same here
        # ball_pos = torch.tensor(self.obs[8:11])  # ensure get_observations has ball_pos set in this column
        # ball_vel = torch.tensor(self.obs[11:])
        left_finger_pose = dc.get_rigid_body_pose(dc.get_rigid_body(self._kinova.prim_path + "/robotiq_85_left_finger_tip_link"))
        right_finger_pose = dc.get_rigid_body_pose(dc.get_rigid_body(self._kinova.prim_path + "/robotiq_85_right_finger_tip_link"))
        # gripper_pos = torch.tensor(0.5*(np.array(left_finger_pose.p) + np.array(right_finger_pose.p)))
        gripper_pos = torch.tensor(0.5 * (np.array(left_finger_pose.p) + np.array(right_finger_pose.p)), dtype=torch.float64)
        ball_gripper_dist = torch.norm(gripper_pos - ball_pos)



        # print(f"is_done | ball_pos: {ball_pos} | height: {ball_pos[2]} | ball_vel_norm: {torch.norm(ball_vel)}")

        # TODO: Handle if the robot catches the ball... Done

        # If centroid of ball is below 10 cm, end episode
        resets = torch.where(ball_pos[2] < 0.1, torch.tensor(1), torch.tensor(0))

        # reset the robot if the ball is within a 10 cm of the grippers centroid and near stationary
        # combined_condition = (ball_gripper_dist < 0.10) & (torch.norm(ball_vel) < 0.01)
        # resets = torch.where(combined_condition, torch.tensor(1), resets)

        # print(f"Resets: {resets}")
        self.resets = resets

        return resets.item() or self._num_frames > self._max_episode_length
