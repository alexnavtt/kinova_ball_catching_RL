import math

import numpy as np
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

dc = _dynamic_control.acquire_dynamic_control_interface()

class KinovaTask(BaseTask):
    # Alex (Done)
    def __init__(self, name, offset=None):
        # self.update_config(sim_config)
        self._max_episode_length = 500

        # Task-specific parameters (fill in as you need them)
        self._robot_lower_joint_limits  = np.deg2rad([-180.0, -128.9, -180.0, -147.8, -180.0, -120.3, -180.0], dtype=np.float32)
        self._robot_upper_joint_limits  = -1.0 * self._robot_lower_joint_limits
        self._gripper_lower_joint_limit = np.deg2rad([ 0.0], dtype=np.float32)
        self._gripper_upper_joint_limit = np.deg2rad([46.0], dtype=np.float32)

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

    # Caleb (used for set up)
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
    
    # Caleb
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
            self._ball = scene.add(
                DynamicSphere(
                    prim_path="/World/random_sphere",
                    name="tennis_ball",
                    position=np.array([0.3, 0.3, 0.3]),
                    scale=np.array([0.0515, 0.0515, 0.0515]),
                    color=np.array([0, 1.0, 0]),
                )
            )

            # set default camera viewport position and target
            self.set_initial_camera_params()

            return

    # Caleb (used for set up)
    def get_kinova(self):
        kinova = Kinova(prim_path="/World/envs/kinova", name="kinova")
        # self._sim_config.apply_articulation_settings(
        #     "Kinova", get_prim_at_path(kinova.prim_path), self._sim_config.parse_actor_config("Kinova")
        # )

    # Crasun
    def set_initial_camera_params(
        self, camera_position=[10, 10, 3], camera_target=[0, 0, 0]
    ):
        set_camera_view(
            eye=camera_position,
            target=camera_target,
            camera_prim_path="/OmniverseKit_Persp",
        )  # need to test this in GUI

    # Alex
    def post_reset(self):
        """
        This gets executed once the scene is constructed and simulation starts running
        """
        self._gripper_dof_index_1 = self._kinovas.get_dof_index("robotiq_85_left_knuckle_joint")
        self._gripper_dof_index_2 = self._kinovas.get_dof_index("robotiq_85_right_knuckle_joint")

        # randomize all envs
        # indices = torch.arange(
        #     self._robots.count, dtype=torch.int64, device=self._device
        # )
        # self.reset(indices)
        #TODO 
        #and set the velocity of the ball

    # Caleb
    def reset(self, env_ids=None):
        """
        This method is used to set our environment into an initial state for starting a new training episode
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self._device)
        num_resets = len(env_ids)

        # zero DOF positions and velocities
        dof_pos = torch.zeros((num_resets, self._kinovas.num_dof), device=self._device)
        dof_vel = torch.zeros((num_resets, self._kinovas.num_dof), device=self._device)

        # reset configuration of the robot
        indices = [0]
        self._kinovas.set_joint_positions(dof_pos, indices=indices)
        self._kinovas.set_joint_velocities(dof_vel, indices=indices)

        # reset configuration of the ball
        self._ball.set_world_pose([0.3, 0.3, 0.3])
        dc.set_rigid_body_linear_velocity(dc.get_rigid_body(self._ball.prim_path), [0.0, 0.0, 5.0])

        # bookkeeping
        # self.resets[env_ids] = 0


    # Crasun
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
        print(f"Joint state actions: {joint_state_actions}")

        full_dof_actions = torch.zeros(13, dtype=torch.float32)
        full_dof_actions[:7] = joint_state_actions[:7]
        full_dof_actions[11] = joint_state_actions[-1]
        full_dof_actions[12] = -1.0 * joint_state_actions[-1]

        # indices = torch.arange(self._cartpoles.count, dtype=torch.int32, device=self._device)
        indices = torch.arange(
            self._kinovas.count, dtype=torch.int32, device=self._device
        )

        # TODO include apply action instead of set joint, double check syntax
        self._kinovas.apply_action(
            ArticulationAction(
                joint_positions=full_dof_actions,
            ),
            indices=indices
        )
        # References
        # https://docs.omniverse.nvidia.com/isaacsim/latest/isaac_gym_tutorials/tutorial_gym_isaac_gym_new_oige_example.html#isaac-sim-app-tutorial-gym-omni-isaac-gym-new-example
        # https://docs.omniverse.nvidia.com/py/isaacsim/source/extensions/omni.isaac.manipulators/docs/index.html

    # Alex
    def get_observations(self):
        # Get the robot and ball state from the simulation
        dof_pos  = self._kinovas.get_joint_positions()
        dof_ball = self._ball.get_world_pose()
        vel_ball = dc.get_rigid_body_linear_velocity(dc.get_rigid_body(self._ball.prim_path))

        print(f"DoF ball: {dof_ball}\nDoF pos: {dof_pos}\nVel ball: {vel_ball}")

        # Extract the information that we need for the model
        joint_pos   = dof_pos[0, 0:7]
        gripper_pos = dof_pos[0, self._gripper_dof_index_1],
        ball_pos    = dof_ball[0][0:3]
        ball_vel    = vel_ball
        #TODO include ball velocity

        print(f"Ball pos: {ball_pos}\nJoint pos: {joint_pos}\nGripper pos: {gripper_pos}\nBall vel: {ball_vel}")

        # # populate the observations buffer                               #|
        # self.obs_buf[:, 0] = joint_pos                                   #|I added these to work with the calculate_metrics method
        # self.obs_buf[:, 1] = gripper_pos                                 #|to reflect what the tutorials do - Caleb
        # self.obs_buf[:, 2] = ball_pos                                    #|
        # # construct the observations dictionary and return               #|
        # observations = {self._cartpoles.name: {"obs_buf": self.obs_buf}} #|
    
        obs = np.concatenate((joint_pos, gripper_pos, ball_pos, ball_vel))
        self.obs = torch.tensor(obs)
        print(f"Self.obs: {self.obs}")
        return self.obs
    
    # Caleb
    def calculate_metrics(self) -> None:
        # use states from the observation buffer to compute reward
        # TODO: Alter this for multiple robots
        joint_pos = self.obs[:7]
        gripper_pos = self.obs[7]
        ball_pos = self.obs[8:]

        # define the reward function based on the end effector colliding with the ball
        # reward = torch.where(torch.abs(gripper_pos) == ball_pos, torch.ones_like(reward) * 2.0, reward)

        # define the reward function based on the end effector distance to the ball
        # first calculate euclidean dist between ball and gripper
        # ball_gripper_dist=math.sqrt((gripper_pos[0]-ball_pos[0])**2+(gripper_pos[1]-ball_pos[1])**2+(gripper_pos[2]-ball_pos[2])**2)
        # make the reward inversly proportional to ball to gripper distance
        # reward = 1.0 /(1+ball_gripper_dist)
        
        # # define the reward function based on the end effector not dropping the ball(hold of on this one probably)
        # reward = torch.where(torch.abs(gripper_pos) == ball_pos, torch.ones_like(reward) * 2.0, reward)
        
        # # penalize the policy if the end effector drops the ball 
        # reward = torch.where(torch.abs(cart_pos) > self._reset_dist, torch.ones_like(reward) * -2.0, reward)


        # # assign rewards to the reward buffer
        # self.rew_buf[:] = reward
        # return reward.item()
        return 0.0


    # Crasun
    def is_done(self) -> None:
        # cart_pos = self.obs[:, 0]
        # pole_pos = self.obs[:, 2]
        ball_pos = torch.tensor(self.obs[8:11])  # ensure get_observations has ball_pos set in this column
        print(f"is_done | ball_pos: {ball_pos} | height: {ball_pos[2]}")

        # TODO: Handle if the robot catches the ball

        # If centroid? of ball is below 10 cm, end episode
        resets = torch.where(ball_pos[2] < 0.1, 1, 0)  
        print(f"Resets: {resets}")
        self.resets = resets

        return resets.item()
