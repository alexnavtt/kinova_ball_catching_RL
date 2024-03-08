# import numpy as np

# from omni.isaac.core.objects import DynamicSphere
# from omni.isaac.examples.base_sample import BaseSample


# class HelloWorld(BaseSample):
#     def __init__(self) -> None:
#         super().__init__()
#         return

#     def setup_scene(self):
#         world = self.get_world()
#         world.scene.add_default_ground_plane()
#         fancy_cube = world.scene.add(
#             # DynamicCuboid(
#             #     prim_path="/World/random_cube", # The prim path of the cube in the USD stage
#             #     name="fancy_cube", # The unique name used to retrieve the object from the scene later on
#             #     position=np.array([0, 0, 1.0]), # Using the current stage units which is in meters by default.
#             #     scale=np.array([0.5015, 0.5015, 0.5015]), # most arguments accept mainly numpy arrays.
#             #     color=np.array([0, 0, 1.0]), # RGB channels, going from 0-1
#             # )
#             DynamicSphere(
#                 prim_path="/World/random_cube",  # The prim path of the cube in the USD stage
#                 name="fancy_cube",  # The unique name used to retrieve the object from the scene later on
#                 position=np.array(
#                     [0, 0, 5.0]
#                 ),  # Using the current stage units which is in meters by default.
#                 scale=np.array(
#                     [0.5015, 0.5015, 0.5015]
#                 ),  # most arguments accept mainly numpy arrays.
#                 color=np.array([0, 1.0, 0]),  # RGB channels, going from 0-1
#             )
#         )
#         return

#     async def setup_post_load(self):
#         self._world = self.get_world()
#         self._cube = self._world.scene.get_object("fancy_cube")
#         self._world.add_physics_callback(
#             "sim_step", callback_fn=self.print_cube_info
#         )  # callback names have to be unique
#         return


import numpy as np

# from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.core.objects import DynamicSphere
#     def print_cube_info(self, step_size):
#         position, orientation = self._cube.get_world_pose()
from omni.isaac.examples.base_sample import BaseSample
#     # here we define the physics callback to be called before each physics step, all physics callbacks must take
#     # step_size as an argument
from omni.isaac.franka import Franka
from omni.isaac.franka.controllers import PickPlaceController


class HelloWorld(BaseSample):
    def __init__(self) -> None:
        super().__init__()
        return

    def setup_scene(self):
        world = self.get_world()
        world.scene.add_default_ground_plane()
        franka = world.scene.add(
            Franka(prim_path="/World/Fancy_Franka", name="fancy_franka")
        )
        world.scene.add(
            DynamicSphere(
                prim_path="/World/random_cube",
                name="fancy_cube",
                position=np.array([0.3, 0.3, 3]),
                scale=np.array([0.0515, 0.0515, 0.0515]),
                color=np.array([0, 1.0, 0]),
            )
        )
        return

    async def setup_post_load(self):
        self._world = self.get_world()
        self._franka = self._world.scene.get_object("fancy_franka")
        self._fancy_cube = self._world.scene.get_object("fancy_cube")
        # Initialize a pick and place controller
        self._controller = PickPlaceController(
            name="pick_place_controller",
            gripper=self._franka.gripper,
            robot_articulation=self._franka,
        )
        self._world.add_physics_callback("sim_step", callback_fn=self.physics_step)
        # World has pause, stop, play..etc
        # Note: if async version exists, use it in any async function is this workflow
        self._franka.gripper.set_joint_positions(
            self._franka.gripper.joint_opened_positions
        )
        await self._world.play_async()
        return

    # This function is called after Reset button is pressed
    # Resetting anything in the world should happen here
    async def setup_post_reset(self):
        self._controller.reset()
        self._franka.gripper.set_joint_positions(
            self._franka.gripper.joint_opened_positions
        )
        await self._world.play_async()
        return

    def physics_step(self, step_size):
        cube_position, cube_orientation = self._fancy_cube.get_world_pose()
        goal_position = np.array([-0.3, -0.3, 0.0515 / 2.0])
        current_joint_positions = self._franka.get_joint_positions()
        actions = self._controller.forward(
            picking_position=cube_position,
            placing_position=goal_position,
            current_joint_positions=current_joint_positions,
        )
        self._franka.apply_action(actions)
        # Only for the pick and place controller, indicating if the state
        # machine reached the final state.
        if self._controller.is_done():
            self._world.pause()

        # Print info as well
        linear_velocity = self._fancy_cube.get_linear_velocity()
        # will be shown on terminal
        print("Cube position is : " + str(cube_position))
        print("Cube's orientation is : " + str(cube_orientation))
        print("Cube's linear velocity is : " + str(linear_velocity))

        return  # linear_velocity = self._cube.get_linear_velocity()

