
import math
from typing import Optional

import numpy as np
import torch
from omni.isaac.core.robots.robot import Robot
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.stage import add_reference_to_stage
from omniisaacgymenvs.tasks.utils.usd_utils import set_drive
from pxr import PhysxSchema



class Kinova(Robot):
    def __init__(
        self,
        prim_path: str,
        name: Optional[str] = "kinova",
        # usd_path: Optional[str] = None,
        usd_path: Optional[str] = "/home/caleb/Research/kinova_ball_catching_RL/models/kinova_closed_loop.usd",
        translation: Optional[torch.tensor] = None,
        orientation: Optional[torch.tensor] = None,
    ) -> None:
        """[summary]"""

        self._usd_path = usd_path
        self._name = name

        self._position = torch.tensor([1.0, 0.0, 0.0]) if translation is None else translation
        self._orientation = torch.tensor([0.0, 0.0, 0.0, 1.0]) if orientation is None else orientation

        if self._usd_path is None:
            assets_root_path = get_assets_root_path()
            if assets_root_path is None:
                carb.log_error("Could not find Isaac Sim assets folder")
            self._usd_path = assets_root_path + "/Isaac/Robots/Franka/franka_instanceable.usd"

        add_reference_to_stage(self._usd_path, prim_path)

        super().__init__(
            prim_path=prim_path,
            name=name,
            translation=self._position,
            orientation=self._orientation,
            articulation_controller=None,
        )

        dof_paths = [
            # "world/base_joint",
            "base_link/joint_1",
            "shoulder_link/joint_2",
            "half_arm_1_link/joint_3",
            "half_arm_2_link/joint_4",
            "forearm_link/joint_5",
            "spherical_wrist_1_link/joint_6",
            "spherical_wrist_2_link/joint_7",
            # "bracelet_link/end_effector",
            # "end_effector_link/robotiq_85_base_joint",
            "robotiq_85_base_link/robotiq_85_left_inner_knuckle_joint",
            "robotiq_85_base_link/robotiq_85_left_knuckle_joint",
            "robotiq_85_base_link/robotiq_85_right_inner_knuckle_joint",
            "robotiq_85_base_link/robotiq_85_right_knuckle_joint",
            # "robotiq_85_left_knuckle_link/robotiq_85_left_finger_joint",
            "robotiq_85_left_finger_link/robotiq_85_left_finger_tip_joint",
            "robotiq_85_left_finger_tip_link/left_closed_loop_joint",
            # "robotiq_85_right_knuckle_link/robotiq_85_right_finger_joint",
            "robotiq_85_right_finger_link/robotiq_85_right_finger_tip_joint",
            "robotiq_85_right_finger_tip_link/right_closed_loop_joint",

        ]

        drive_type = ["angular"] * 15 # + ["linear"] * 2
        default_dof_pos = [math.degrees(x) for x in [0.0, -1.0, 0.0, -2.2, 0.0, 2.4, 0.8, 0, 0, 0, 0, 0, 0, 0, 0]] #+ [0.02, 0.02]
        stiffness = [400 * np.pi / 180] * 15 #+ [10000] * 2
        damping = [80 * np.pi / 180] * 15 #+ [100] * 2
        max_force = [87, 87, 87, 87, 12, 12, 12, 87, 87, 87, 87, 87, 87, 87, 87]#200, 200]
        max_velocity = [math.degrees(x) for x in [2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61, 2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61, 2.61]] #+ [0.2, 0.2]

        for i, dof in enumerate(dof_paths):
            set_drive(
                prim_path=f"{self.prim_path}/{dof}",
                drive_type=drive_type[i],
                target_type="position",
                target_value=default_dof_pos[i],
                stiffness=stiffness[i],
                damping=damping[i],
                max_force=max_force[i],
            )

            PhysxSchema.PhysxJointAPI(get_prim_at_path(f"{self.prim_path}/{dof}")).CreateMaxJointVelocityAttr().Set(
                max_velocity[i]
            )

    def set_kinova_properties(self, stage, prim):
        for link_prim in prim.GetChildren():
            if link_prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI): 
                rb = PhysxSchema.PhysxRigidBodyAPI.Get(stage, link_prim.GetPrimPath())
                rb.GetDisableGravityAttr().Set(True)

