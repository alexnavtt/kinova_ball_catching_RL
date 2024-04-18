
import math
from typing import Optional

import numpy as np
import torch
import carb
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
        # usd_path: Optional[str] = "/home/alex/workspaces/cs395T/src/kinova_ball_catching_RL/models/kinova_closed_loop.usd", # Alex
        usd_path: Optional[str] = "/home/caleb/Research/kinova_ball_catching_RL/models/kinova_closed_loop.usd", # Caleb
        # usd_path: Optional[str] = "/home/crasun/isaacsim_ws/abc_ws/kinova_ball_catching_RL/models/kinova_closed_loop.usd", # Crasun
        translation: Optional[torch.tensor] = None,
        orientation: Optional[torch.tensor] = None,
    ) -> None:
        """[summary]"""

        self._usd_path = usd_path
        self._name = name

        self._position = torch.tensor([1.0, 0.0, 0.0]) if translation is None else translation
        self._orientation = torch.tensor([0.0, 0.0, 0.0, 1.0]) if orientation is None else orientation

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

        PhysxSchema.PhysxJointAPI(get_prim_at_path(f"{self.prim_path}/robotiq_85_base_link/robotiq_85_left_knuckle_joint")).CreateMaxJointVelocityAttr().Set(
            100.0
        )

        PhysxSchema.PhysxJointAPI(get_prim_at_path(f"{self.prim_path}/robotiq_85_base_link/robotiq_85_right_knuckle_joint")).CreateMaxJointVelocityAttr().Set(
            100.0
        )

    def set_kinova_properties(self, stage, prim):
        for link_prim in prim.GetChildren():
            if link_prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
                rb = PhysxSchema.PhysxRigidBodyAPI.Get(stage, link_prim.GetPrimPath())
                rb.GetDisableGravityAttr().Set(True)
