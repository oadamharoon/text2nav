import omni
import numpy as np
from isaacsim import SimulationApp
_simulation_app = SimulationApp(launch_config={"headless": True, "anti_aliasing": 0})
print("Simulation app initialized")
from omni.isaac.sensor import Camera
import omni.isaac.core.utils.numpy.rotations as rot_utils
import numpy as np
import omni
import cv2
from omni.isaac.core import PhysicsContext
import os

from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.wheeled_robots.robots import WheeledRobot
# This extension includes several generic controllers that could be used with multiple robots
from omni.isaac.wheeled_robots.controllers.wheel_base_pose_controller import WheelBasePoseController
# Robot specific controller
from omni.isaac.wheeled_robots.controllers.differential_controller import DifferentialController
import numpy as np
from omni.isaac.core import World
from pxr import UsdGeom, Gf, PhysxSchema, UsdPhysics, Sdf, Usd


class HelloWorld:
    def __init__(self) -> None:
        return

    def setup_scene(self):
        config={"physics_dt": 1/60, "rendering_dt": 1/60}
        self.world = World(physics_dt=config["physics_dt"],
                            rendering_dt=config["rendering_dt"],
                            stage_units_in_meters=1.0,
                            set_defaults=False,
                            backend="torch",
                            device="cuda")
        
        self.physics_context = PhysicsContext()
        self.physics_context.set_solver_type("TGS")
        self.physics_context.set_broadphase_type("GPU")
        self.physics_context.enable_gpu_dynamics(True)

        self.world.scene.add_default_ground_plane()
        stage = omni.usd.get_context().get_stage()
        assets_root_path = get_assets_root_path()
        jetbot_asset_path = assets_root_path + "/Isaac/Robots/Jetbot/jetbot.usd"
        self._jetbot = self.world.scene.add(
            WheeledRobot(
                prim_path="/World/Fancy_Robot2",
                # name="fancy_robot2",
                wheel_dof_names=["left_wheel_joint", "right_wheel_joint"],
                create_robot=True,
                usd_path=jetbot_asset_path,
            )
        )
        self.setup_objects(stage)
        self.post_setup()
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
        self.translate_op = next((op for op in parent_xform_ops if op.GetOpType() == UsdGeom.XformOp.TypeTranslate), None)
        rotate_op = next((op for op in parent_xform_ops if op.GetOpType() == UsdGeom.XformOp.TypeRotateXYZ), None)
        
        if not self.translate_op:
            self.translate_op = parent_xform.AddTranslateOp()
        if not rotate_op:
            rotate_op = parent_xform.AddRotateXYZOp()
            
        self.translate_op.Set(Gf.Vec3d(0.1, -0.125, 0.3))
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

    def post_setup(self):
        self._world = self.world
        self._jetbot = self._world.scene.get_object("fancy_robot2")
        # self._world.add_physics_callback("sending_actions", callback_fn=self.send_robot_actions)
        # Initialize our controller after load and the first reset
        self._my_controller = WheelBasePoseController(name="cool_controller",
                                                        open_loop_wheel_controller=
                                                            DifferentialController(name="simple_control",
                                                                                    wheel_radius=0.03, wheel_base=0.1125), is_holonomic=False)
        return

    def send_robot_actions(self, goal_position):
        position, orientation = self._jetbot.get_world_pose()
        self._jetbot.apply_action(self._my_controller.forward(start_position=position,
                                                            start_orientation=orientation,
                                                            goal_position=goal_position))
        self.goal_position = goal_position
        return
    
    def _setup_camera(self):
        """Setup the camera with appropriate parameters"""
        calculated_dynamic_frequency = int(1 / (self.texture_update_interval))
        self.camera = Camera(
            prim_path="/World/Fancy_Robot2/Camera",
            position=np.array([0.0, 0.0, 0.4]),
            frequency=calculated_dynamic_frequency,
            resolution=(256, 256),
        )
        # Add camera to the world
        self.camera.initialize()
    
    def get_observations(self):
        frame = self.camera.get_current_frame()
        if frame is not None and 'rgba' in frame:
            rgba_image = frame['rgba']
            if rgba_image.dtype != np.uint8:
                rgba_image = rgba_image.astype(np.uint8)
            
            rgb_image = cv2.cvtColor(rgba_image, cv2.COLOR_RGBA2RGB)

    def distance2goal(self):
        """Calculate the distance to the goal position"""
        current_position, _ = self._jetbot.get_world_pose()
        distance = np.linalg.norm(np.array(current_position) - np.array(self.goal_position))
        return distance
    
    def change_object_location(self, new_position):
        """Change the location of the object in the scene"""
        self.translate_op.Set(value=Gf.Vec3d(new_position))
        return
