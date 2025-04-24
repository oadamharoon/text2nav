import omni
from omni.isaac.examples.base_sample import BaseSample
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.wheeled_robots.robots import WheeledRobot
# This extension includes several generic controllers that could be used with multiple robots
from omni.isaac.wheeled_robots.controllers.wheel_base_pose_controller import WheelBasePoseController
# Robot specific controller
from omni.isaac.wheeled_robots.controllers.differential_controller import DifferentialController
from omni.isaac.sensor import Camera
from pxr import Usd, UsdLux, Gf, Sdf, UsdGeom, UsdShade, Vt, UsdPhysics
from omni.isaac.core import SimulationContext
import numpy as np
import os
import cv2
from datetime import datetime

class HelloWorld(BaseSample):
    def __init__(self) -> None:
        super().__init__()
        self.simulation_context = SimulationContext(set_defaults=True)
        self._goal_position = np.array([0.8, 0.8])
        self._goal_threshold = 0.06
        self._episode_folder = None
        return

    def setup_scene(self):
        world = self.get_world()
        world.scene.add_default_ground_plane()
        stage = omni.usd.get_context().get_stage()
        assets_root_path = get_assets_root_path()
        jetbot_asset_path = assets_root_path + "/Isaac/Robots/Jetbot/jetbot.usd"
        world.scene.add(
            WheeledRobot(
                prim_path="/World/Fancy_Robot",
                name="fancy_robot",
                wheel_dof_names=["left_wheel_joint", "right_wheel_joint"],
                create_robot=True,
                usd_path=jetbot_asset_path,
            )
        )
        self._setup_camera()
        self.setup_objects(stage)
        return
    
    def _setup_camera(self):
        """Setup the camera with appropriate parameters"""
        self.camera = Camera(
            prim_path="/World/Fancy_Robot/chassis/rgb_camera/jetbot_camera",
            name="jetbot_camera",
            resolution=(256, 256)
        )
        # Add camera to the world
        self.camera.initialize()

        # Wait for camera to be ready
        self._camera_ready = False

        # Variables for controlling image capture
        self._last_capture_time = 0.0
        self._image_counter = 1
        self._capture_interval = 0.5  # 1 second between captures

        # Create episode folder
        self._create_episode_folder()
    
    def _create_episode_folder(self):
        """Create a unique folder for each episode based on date and time"""
        base_dir = "C:/Users/oadam/Downloads/text2nav_trajectory"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._episode_folder = os.path.join(base_dir, f"episode_{timestamp}")
        os.makedirs(self._episode_folder, exist_ok=True)
        print(f"Created episode folder: {self._episode_folder}")
    
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
            
        self.translate_op.Set(Gf.Vec3d(1.1, 1.1, 0.075))
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
        # Initialize our controller after load and the first reset
        self._my_controller = WheelBasePoseController(name="cool_controller",
                                                        open_loop_wheel_controller=
                                                            DifferentialController(name="simple_control",
                                                                                   wheel_radius=0.03,
                                                                                   wheel_base=0.1125),
                                                    is_holonomic=False)
        # Add camera callback only after ensuring camera is ready
        self._world.add_physics_callback("camera_check", callback_fn=self._check_camera_ready)
        return

    def send_robot_actions(self, step_size):
        position, orientation = self._jetbot.get_world_pose()
        # Check if we're within the goal threshold
        distance_to_goal = np.linalg.norm(position[:2] - self._goal_position)
        if distance_to_goal < self._goal_threshold:
            print(f"Goal reached! Distance: {distance_to_goal:.3f}m")
            self.simulation_context.stop()
            return
        self._jetbot.apply_action(self._my_controller.forward(start_position=position,
                                                            start_orientation=orientation,
                                                            goal_position=self._goal_position))
        return

    def _check_camera_ready(self, step_size):
        """Check if camera is ready and then add capture callback"""
        if not self._camera_ready:
            try:
                # Test if camera can get a frame
                frame = self.camera.get_current_frame()
                if frame is not None and 'rgba' in frame:
                    self._camera_ready = True
                    # Now add the capture callback
                    self._world.add_physics_callback("capture_camera_frame", callback_fn=self.capture_camera_frame)
                    # Remove this check callback
                    self._world.remove_physics_callback("camera_check")
            except:
                pass
        return

    def capture_camera_frame(self, step_size):
        """Capture the camera frame from the camera object"""
        try:
            # Check if camera exists and is initialized
            if not hasattr(self, 'camera') or self.camera is None:
                return
            
            # Get current time
            current_time = self._world.current_time
            
            # Check if enough time has passed since last capture
            if current_time - self._last_capture_time < self._capture_interval:
                return
            
            save_directory = "C:/Users/oadam/Downloads/text2nav_trajectory"
            os.makedirs(save_directory, exist_ok=True)
            
            frame = self.camera.get_current_frame()
            
            # Validate frame data
            if frame is None or 'rgba' not in frame:
                return
            
            rgba_image = frame['rgba']
            if rgba_image is None or rgba_image.size == 0:
                return
            
            # Convert to uint8 if needed
            if rgba_image.dtype != np.uint8:
                rgba_image = rgba_image.astype(np.uint8)
            
            # Convert to RGB
            rgb_image = cv2.cvtColor(rgba_image, cv2.COLOR_RGBA2RGB)
            
            # Save image with sequential naming
            filename = os.path.join(self._episode_folder, f"N_{self._image_counter}.png")
            cv2.imwrite(filename, rgb_image)
            
            # Update counters
            self._last_capture_time = current_time
            self._image_counter += 1
            
            print(f"Saved image: {filename}")
            
        except Exception as e:
            print(f"Error capturing camera frame: {e}")
        
        return
