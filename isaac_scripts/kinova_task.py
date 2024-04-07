import math

import numpy as np
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.tasks.base_task import BaseTask
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.prims import create_prim
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.viewports import set_camera_view

import torch
from gymnasium import spaces


class KinovaTask(BaseTask):
    # Alex
    def __init__(self, name, offset=None):
        # task-specific parameters
        # TODO

        # values used for defining RL buffers
        self._num_observations = 4
        self._num_actions = 1
        self._device = "cpu"
        self.num_envs = 1

        # a few class buffers to store RL-related states
        self.obs = torch.zeros((self.num_envs, self._num_observations))
        self.resets = torch.zeros((self.num_envs, 1))

        # set the action and observation space for RL (lower and upper limits)
        self.action_space = spaces.Box(
            np.ones(self._num_actions, dtype=np.float32) * -1.0,
            np.ones(self._num_actions, dtype=np.float32) * 1.0,
        )
        self.observation_space = spaces.Box(
            np.ones(self._num_observations, dtype=np.float32) * -np.Inf,
            np.ones(self._num_observations, dtype=np.float32) * np.Inf,
        )

        # trigger __init__ of parent class
        BaseTask.__init__(self, name=name, offset=offset)

    # Caleb
    def set_up_scene(self, scene) -> None:
        # Add the kinova to the stage
        # prim_path = "/World/kinova"
        # add_reference_to_stage(usd_path="/home/alex/workspaces/cs395T/src/kinova_ball_catching_RL/models/kinova_closed_loop.usd", prim_path=prim_path)

        # create an ArticulationView wrapper for our cartpole - this can be extended towards accessing multiple cartpoles
        self._robots = ArticulationView(
            prim_paths_expr="/World/kinova*", name="kinova_view"
        )

        # add Cartpole ArticulationView and ground plane to the Scene
        scene.add(self._robots)
        scene.add_default_ground_plane()

        # Add and launch the ball
        self._ball = world.scene.add(
            DynamicSphere(
                prim_path="/World/random_sphere",
                name="tennis_ball",
                position=np.array([0.3, 0.3, 0.3]),
                scale=np.array([0.0515, 0.0515, 0.0515]),
                color=np.array([0, 1.0, 0]),
                linear_velocity=np.array([1, -1, 0]),
            )
        )

        # set default camera viewport position and target
        self.set_initial_camera_params()

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
        self._cart_dof_idx = self._robots.get_dof_index("cartJoint")
        self._pole_dof_idx = self._robots.get_dof_index("poleJoint")

        # randomize all envs
        indices = torch.arange(
            self._robots.count, dtype=torch.int64, device=self._device
        )
        self.reset(indices)

    # Caleb
    def reset(self, env_ids=None):
        """
        This method is used to set our environment into an initial state for starting a new training episode
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self._device)
        num_resets = len(env_ids)

        # randomize DOF positions
        dof_pos = torch.zeros(
            (num_resets, self._cartpoles.num_dof), device=self._device
        )
        dof_pos[:, self._cart_dof_idx] = 1.0 * (
            1.0 - 2.0 * torch.rand(num_resets, device=self._device)
        )
        dof_pos[:, self._pole_dof_idx] = (
            0.125 * math.pi * (1.0 - 2.0 * torch.rand(num_resets, device=self._device))
        )

        # randomize DOF velocities
        dof_vel = torch.zeros(
            (num_resets, self._cartpoles.num_dof), device=self._device
        )
        dof_vel[:, self._cart_dof_idx] = 0.5 * (
            1.0 - 2.0 * torch.rand(num_resets, device=self._device)
        )
        dof_vel[:, self._pole_dof_idx] = (
            0.25 * math.pi * (1.0 - 2.0 * torch.rand(num_resets, device=self._device))
        )

        # apply resets
        indices = env_ids.to(dtype=torch.int32)
        self._cartpoles.set_joint_positions(dof_pos, indices=indices)
        self._cartpoles.set_joint_velocities(dof_vel, indices=indices)

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
        # References
        # https://docs.omniverse.nvidia.com/isaacsim/latest/isaac_gym_tutorials/tutorial_gym_isaac_gym_new_oige_example.html#isaac-sim-app-tutorial-gym-omni-isaac-gym-new-example
        # https://docs.omniverse.nvidia.com/py/isaacsim/source/extensions/omni.isaac.manipulators/docs/index.html

    # Alex
    def get_observations(self):
        dof_pos = self._robots.get_joint_positions()
        dof_vel = self._robots.get_joint_velocities()

        # collect pole and cart joint positions and velocities for observation
        cart_pos = dof_pos[:, self._cart_dof_idx]
        cart_vel = dof_vel[:, self._cart_dof_idx]
        pole_pos = dof_pos[:, self._pole_dof_idx]
        pole_vel = dof_vel[:, self._pole_dof_idx]

        self.obs[:, 0] = cart_pos
        self.obs[:, 1] = cart_vel
        self.obs[:, 2] = pole_pos
        self.obs[:, 3] = pole_vel

        return self.obs

    # Caleb
    def calculate_metrics(self) -> None:
        cart_pos = self.obs[:, 0]
        cart_vel = self.obs[:, 1]
        pole_angle = self.obs[:, 2]
        pole_vel = self.obs[:, 3]

        # compute reward based on angle of pole and cart velocity
        reward = (
            1.0
            - pole_angle * pole_angle
            - 0.01 * torch.abs(cart_vel)
            - 0.005 * torch.abs(pole_vel)
        )
        # apply a penalty if cart is too far from center
        reward = torch.where(
            torch.abs(cart_pos) > self._reset_dist,
            torch.ones_like(reward) * -2.0,
            reward,
        )
        # apply a penalty if pole is too far from upright
        reward = torch.where(
            torch.abs(pole_angle) > np.pi / 2, torch.ones_like(reward) * -2.0, reward
        )

        return reward.item()

    # Crasun
    def is_done(self) -> None:
        # cart_pos = self.obs[:, 0]
        # pole_pos = self.obs[:, 2]
        ball_pos = self.obs[
            :, 0
        ]  # ensure get_observations has ball_pos set in this column

        gripper_pos = self.obs[
            :, 1
        ]  # assuming gripper cartesian position is in this column. Might need FK if using joint space

        # reset the robot if cart has reached reset_dist or pole is too far from upright
        # resets = torch.where(torch.abs(cart_pos) > self._reset_dist, 1, 0)
        # resets = torch.where(torch.abs(pole_pos) > math.pi / 2, 1, resets)
        resets = torch.where(
            ball_pos[0] - gripper_pos[0] < -0.01
        )  # assuming we launch the ball such that the ball's x coodinate is greater than the gripper's.
        self.resets = resets

        return resets.item()