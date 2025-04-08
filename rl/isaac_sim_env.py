import omni
from omni.isaac.examples.base_sample import BaseSample
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.wheeled_robots.robots import WheeledRobot
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core import SimulationContext
import numpy as np
from pxr import Usd, UsdGeom, Gf

from omni.isaac.examples.base_sample import BaseSample
from omni.isaac.core.world import World
from omni.isaac.core.materials import OmniPBR
from omni.isaac.core import SimulationContext
from omni.isaac.examples.user_examples.CalibrationBoardGenerator import CalibrationBoardGenerator
from omni.isaac.sensor import Camera
import omni.isaac.core.utils.numpy.rotations as rot_utils
import numpy as np
import omni
from pxr import Usd, UsdLux, Gf, Sdf, UsdGeom, UsdShade, Vt, UsdPhysics
import cv2
import os
import re
import time
import carb

class HelloWorld(BaseSample):
    def __init__(self) -> None:
        super().__init__()
        return

    def setup_scene(self):
        world = self.get_world()
        world.scene.add_default_ground_plane()
        stage = omni.usd.get_context().get_stage()
        assets_root_path = get_assets_root_path()
        jetbot_asset_path = assets_root_path + "/Isaac/Robots/Jetbot/jetbot.usd"
        self._jetbot = world.scene.add(
            WheeledRobot(
                prim_path="/World/Fancy_Robot",
                name="fancy_robot",
                wheel_dof_names=["left_wheel_joint", "right_wheel_joint"],
                create_robot=True,
                usd_path=jetbot_asset_path,
            )
        )
        self.setup_objects(stage)
        return
    
    def setup_objects(self, stage):
        # Create a parent Xform for the scan object
        parent_path = "/World/Object"
        # Define the parent transform
        parent_prim = stage.DefinePrim(parent_path, "Xform")
        
        # Load the YCB pitcher as a reference under the parent
        object_path = f"{parent_path}/pitcher"
        asset_path = "http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.1/Isaac/Props/YCB/Axis_Aligned/019_pitcher_base.usd"
        object_prim = stage.DefinePrim(object_path, "Xform")
        object_prim.GetReferences().AddReference(asset_path)
        
        # Set up transforms for parent (position and rotation)
        parent_xform = UsdGeom.Xformable(parent_prim)
        parent_xform_ops = parent_xform.GetOrderedXformOps()
        
        # Setup parent transform operations
        translate_op = next((op for op in parent_xform_ops if op.GetOpType() == UsdGeom.XformOp.TypeTranslate), None)
        rotate_op = next((op for op in parent_xform_ops if op.GetOpType() == UsdGeom.XformOp.TypeRotateXYZ), None)
        
        if not translate_op:
            translate_op = parent_xform.AddTranslateOp()
        if not rotate_op:
            rotate_op = parent_xform.AddRotateXYZOp()
            
        translate_op.Set(Gf.Vec3d(0.1, -0.125, 1.5))
        rotate_op.Set(Gf.Vec3f(0, 0, 0))  # Initial rotation
        # Set up transforms for the pitcher object 
        object_xform = UsdGeom.Xformable(object_prim)
        object_xform_ops = object_xform.GetOrderedXformOps()
        # Setup object transform operations
        scale_op = next((op for op in object_xform_ops if op.GetOpType() == UsdGeom.XformOp.TypeScale), None)
        if not scale_op:
            scale_op = object_xform.AddScaleOp()
        
        scale_op.Set(Gf.Vec3d(1,1,1))
        return

    async def setup_post_load(self):
        self._world = self.get_world()
        self._jetbot = self._world.scene.get_object("fancy_robot")
        self._world.add_physics_callback("sending_actions", callback_fn=self.send_robot_actions)
        return

    def send_robot_actions(self, step_size):
        self._jetbot.apply_wheel_actions(ArticulationAction(joint_positions=None,
                                                            joint_efforts=None,
                                                            joint_velocities=5 * np.random.rand(2,)))
        return
