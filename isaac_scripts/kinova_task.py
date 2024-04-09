import math

import numpy as np
from omni.isaac.core.articulations import ArticulationView
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

        # Values used for defining RL buffers
        self._num_observations = 14 # 7 robot joint states, 1 gripper joint state and 3 ball states (TODO: Image input)
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
            # np.concatenate((self._robot_lower_joint_limits, self._gripper_lower_joint_limit, -1.0*np.ones((1,3),dtype=np.float32)*np.Inf)),
            # np.concatenate((self._robot_upper_joint_limits, self._gripper_upper_joint_limit, +1.0*np.ones((1,3),dtype=np.float32)*np.Inf))
            np.concatenate((self._robot_lower_joint_limits, self._gripper_lower_joint_limit, -1.0*np.ones(3,dtype=np.float32)*np.Inf)),
            np.concatenate((self._robot_upper_joint_limits, self._gripper_upper_joint_limit, +1.0*np.ones(3,dtype=np.float32)*np.Inf))
            
            #TODO add velocity
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
            # self.get_kinova()

            # call the parent class to clone the single environment
            # super().set_up_scene(scene)

            # construct an ArticulationView object to hold our collection of environments
            self._kinovas = ArticulationView(
                prim_paths_expr="/World/envs/.*/Kinova", name="kinova_view", reset_xform_properties=False
            )
            # register the ArticulationView object to the world, so that it can be initialized
            # scene.add(self._kinovas)

            # # self._kinovas = KinovaView(prim_paths_expr="/World/envs/.*/kinova", name="kinova_view")
            # create_prim(prim_path="/World/kinova", prim_type="Xform", position=self._position)
            # add_reference_to_stage(usd_path="/home/caleb/Research/kinova_ball_catching_RL/models/kinova_closed_loop.usd", "/World/kinova")
            # self._kinovas = KinovaView(prim_paths_expr="/World/kinova", name="kinova_view")

            scene.add_default_ground_plane()
            scene.add(self._kinovas)
            # scene.add(self._frankas._hands)
            # scene.add(self._frankas._lfingers)
            # scene.add(self._frankas._rfingers)

            # Add and launch the ball (this should maybe be moved to post_reset?)
            self._ball = self.scene.add(
                DynamicSphere(
                    prim_path="/World/random_sphere",
                    name="tennis_ball",
                    position=np.array([0.3, 0.3, 0.3]),
                    scale=np.array([0.0515, 0.0515, 0.0515]),
                    color=np.array([0, 1.0, 0]),
                    linear_velocity=np.array([1, -1, 0]),
                )
            )#TODO move to post reset

            # set default camera viewport position and target
            self.set_initial_camera_params()

            return

    # Caleb (used for set up)
    def get_kinova(self):
        kinova = Kinova(prim_path="/home/caleb/Research/kinova_ball_catching_RL/isaac_scripts" + "/kinova", name="kinova")
        self._sim_config.apply_articulation_settings(
            "Kinova", get_prim_at_path(kinova.prim_path), self._sim_config.parse_actor_config("Kinova")
        )

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
        self._gripper_dof_index_1 = self._robots.get_dof_index("robotiq_85_left_knuckle_joint")
        self._gripper_dof_index_2 = self._robots.get_dof_index("robotiq_85_right_knuckle_joint")

        # randomize all envs
        indices = torch.arange(
            self._robots.count, dtype=torch.int64, device=self._device
        )
        self.reset(indices)
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

        # reset props (ball in our case) need to write get_props method
        # if self.num_props > 0:
        #     self._props.set_world_poses(
        #         self.default_prop_pos[self.prop_indices[env_ids].flatten()],
        #         self.default_prop_rot[self.prop_indices[env_ids].flatten()],
        #         self.prop_indices[env_ids].flatten().to(torch.int32),
        #     )

        #TODO reset configuration of the robot
        #and the pose\ of the ball

        # apply resets
        indices = env_ids.to(dtype=torch.int32)
        self._kinovas.set_joint_positions(dof_pos, indices=indices)
        self._kinovas.set_joint_velocities(dof_vel, indices=indices)

        # bookkeeping
        self.resets[env_ids] = 0


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

        # forces = torch.zeros((self._cartpoles.count, self._cartpoles.num_dof), dtype=torch.float32, device=self._device)
        # forces[:, self._cart_dof_idx] = self._max_push_effort * actions[0]

        # indices = torch.arange(self._cartpoles.count, dtype=torch.int32, device=self._device)
        indices = torch.arange(
            self._robots.count, dtype=torch.int32, device=self._device
        )
        # self._cartpoles.set_joint_efforts(forces, indices=indices)
        self._robots.set_joint_positions(positions=joint_state_actions, indices=indices)
        # TODO include apply action instead of set joint, double check syntax
        # References
        # https://docs.omniverse.nvidia.com/isaacsim/latest/isaac_gym_tutorials/tutorial_gym_isaac_gym_new_oige_example.html#isaac-sim-app-tutorial-gym-omni-isaac-gym-new-example
        # https://docs.omniverse.nvidia.com/py/isaacsim/source/extensions/omni.isaac.manipulators/docs/index.html

    # Alex
    def get_observations(self):
        # Get the robot and ball state from the simulation
        dof_pos  = self._robots.get_joint_positions()
        dof_ball = self._ball.get_world_pose()

        # Extract the information that we need for the model
        joint_pos   = dof_pos[:, 0:7]
        gripper_pos = dof_pos[:, self._gripper_dof_index_1] 
        ball_pos    = dof_ball[:, 0:3]
        #TODO include ball velocity

        # # populate the observations buffer                               #|
        # self.obs_buf[:, 0] = joint_pos                                   #|I added these to work with the calculate_metrics method
        # self.obs_buf[:, 1] = gripper_pos                                 #|to reflect what the tutorials do - Caleb
        # self.obs_buf[:, 2] = ball_pos                                    #|
        # # construct the observations dictionary and return               #|
        # observations = {self._cartpoles.name: {"obs_buf": self.obs_buf}} #|
    
        self.obs = np.concatenate((joint_pos, gripper_pos, ball_pos))#,observations
        return self.obs
    
    # Caleb
    def calculate_metrics(self) -> None:
        # use states from the observation buffer to compute reward
        joint_pos = self.obs[:, 0]
        gripper_pos = self.obs[:, 1]
        ball_pos = self.obs[:, 2]

        # define the reward function based on the end effector colliding with the ball
        reward = torch.where(torch.abs(gripper_pos) == ball_pos, torch.ones_like(reward) * 2.0, reward)

        # define the reward function based on the end effector distance to the ball
        # first calculate euclidean dist between ball and gripper
        ball_gripper_dist=math.sqrt((gripper_pos[0]-ball_pos[0])**2+(gripper_pos[1]-ball_pos[1])**2+(gripper_pos[2]-ball_pos[2])**2)
        # make the reward inversly proportional to ball to gripper distance
        reward = 1.0 /(1+ball_gripper_dist)
        
        # # define the reward function based on the end effector not dropping the ball(hold of on this one probably)
        # reward = torch.where(torch.abs(gripper_pos) == ball_pos, torch.ones_like(reward) * 2.0, reward)
        
        # # penalize the policy if the end effector drops the ball 
        # reward = torch.where(torch.abs(cart_pos) > self._reset_dist, torch.ones_like(reward) * -2.0, reward)


        # # assign rewards to the reward buffer
        # self.rew_buf[:] = reward
        return reward.item()


    # Crasun
    def is_done(self) -> None:
        # cart_pos = self.obs[:, 0]
        # pole_pos = self.obs[:, 2]
        ball_pos = self.obs[
            :, 2
        ]  # ensure get_observations has ball_pos set in this column

        # gripper_pos = self.obs[
        #     :, 1
        # ]  # assuming gripper cartesian position is in this column. Might need FK if using joint space

        # reset the robot if cart has reached reset_dist or pole is too far from upright
        # resets = torch.where(torch.abs(cart_pos) > self._reset_dist, 1, 0)
        # resets = torch.where(torch.abs(pole_pos) > math.pi / 2, 1, resets)
        resets = torch.where(
            ball_pos[3]  < .1 #if centroid? of ball is below 10 cm, end episode
        )  # assuming we launch the ball such that the ball's x coodinate is greater than the gripper's.
        self.resets = resets

        return resets.item()
