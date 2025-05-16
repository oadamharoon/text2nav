# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import torch
from collections.abc import Sequence
import omni

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation, ArticulationCfg
from omni.isaac.lab.actuators.actuator_cfg import ImplicitActuatorCfg
from omni.isaac.lab.sensors import TiledCamera, TiledCameraCfg
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationCfg, PhysxCfg
from omni.isaac.lab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from omni.isaac.lab.utils import configclass
import omni.isaac.core.utils.stage as stage_utils
from omni.isaac.core import PhysicsContext
from omni.isaac.lab.utils.math import sample_uniform
from omni.isaac.lab.assets import RigidObject, RigidObjectCfg
from omni.isaac.core.utils.nucleus import get_assets_root_path

import gymnasium as gym
import numpy as np

from d3rlpy.algos import BCConfig
import numpy as np
from .embeddings import EmbeddingPipeline
import csv
import os

import d3rlpy
encoder_factory = d3rlpy.models.VectorEncoderFactory(
        hidden_units=[1024, 512, 256, 128, 64],
        activation='relu',
    )

@configclass
class JetbotCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 20.0
    action_scale = 1.0
    state_space = 0
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 60, render_interval=decimation,
                                       physx=PhysxCfg(gpu_temp_buffer_capacity=2**26),
                                       gravity=(0.0, 0.0, -9.81),
                                    )
    # robot
    assets_root_path = get_assets_root_path()
    robot_cfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Fancy_Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=assets_root_path + "/Isaac/Robots/Jetbot/jetbot.usd",
            activate_contact_sensors=True,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True, solver_position_iteration_count=12, solver_velocity_iteration_count=1
            ),
        ), 
        actuators={
            ".*": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                effort_limit=1000.0,
                velocity_limit=20.0,
                stiffness=800.0,
                damping=4.0,
            ),
        },
    )

    joint_dof_name = ["left_wheel_joint", "right_wheel_joint"]

    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Fancy_Robot/chassis/rgb_camera/Camera",
        offset=TiledCameraCfg.OffsetCfg(pos=(0.0243, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0), convention="world"),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=0.75, 
            focus_distance=0.0, 
            horizontal_aperture=2.35, 
            clipping_range=(0.076, 10.0),
        ),
        width=256,
        height=256,
    )

    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(len(joint_dof_name),))
    # observation_space = {"rgb": gym.spaces.Box(low=0, high=255, shape=(tiled_camera.height, tiled_camera.width, 3), dtype=np.uint8), 
    #                      "poses": gym.spaces.Box(low=-20.00, high=20.0, shape=(10,), dtype=np.float32),
    #                      "goal_index": gym.spaces.Discrete(5)}

    observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)
    
    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=3, replicate_physics=True)

    # reset
    max_distance_to_goal = 15
    maxlen=10
    max_y = 1.45
    min_y = -1.45
    max_x = 2.5
    min_x = -0.2

    # reward scales
    rew_scale_alive = -0.1


class JetbotEnv(DirectRLEnv):
    cfg: JetbotCfg

    def __init__(self, cfg: JetbotCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._joint_dof_idx, _ = self._robot.find_joints(self.cfg.joint_dof_name)
        self.action_scale = self.cfg.action_scale
        self.joint_pos = self._robot.data.joint_pos
        self.joint_vel = self._robot.data.joint_vel
        COLOR_INDEX = {
        "red": 0,
        "green": 1,
        "blue": 2,
        "yellow": 3,
        "pink": 4
        }
        self.INDEX_COLOR = {v: k for k, v in COLOR_INDEX.items()}
        self.rl_model = BCConfig().create(device="cuda:0")
        self.rl_model.create_impl(observation_shape=(1152,), action_size=2)
        model_data = torch.load("/home/nitesh/Downloads/bc_model.pt")
        self.rl_model._impl.policy.load_state_dict(model_data["imitator"])
        self.embedd_pipe = EmbeddingPipeline()

        self.success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.timesteps = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)

    def close(self):
        """Cleanup for the environment."""
        super().close()

    def _setup_scene(self):
        self.stage = stage_utils.get_current_stage()
        self._robot = Articulation(self.cfg.robot_cfg)
        self._tiled_camera = TiledCamera(self.cfg.tiled_camera)
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg(color=(0.67, 0.84, 0.9), size=(1000.0, 1000.0)))
        self.physics_context = PhysicsContext()
        self.physics_context.set_solver_type("TGS")
        self.physics_context.set_broadphase_type("GPU")
        self.physics_context.enable_gpu_dynamics(True)

        # add light
        light_cfg = sim_utils.SphereLightCfg(intensity=1200.0, color=(1.0, 1.0, 1.0), radius=1.98)
        light_cfg.func("/World/envs/env_.*/Light", light_cfg, translation=(1.35, 0.0, 2.0))

        # add background plane
        intensity = 0.6
        background_color = (intensity, intensity, intensity)
        background_plane_cfg = sim_utils.CuboidCfg(size=(0.01, 3.0, 0.5), 
                                                   visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=background_color))
        background_plane_cfg.func("/World/envs/env_.*/BackgroundPlane", background_plane_cfg, translation=(2.7, 0.0, 0.25))

        # add ground plane
        right_plane_cfg = sim_utils.CuboidCfg(size=(4.0, 0.01, 0.5), 
                                                   visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=background_color))
        right_plane_cfg.func("/World/envs/env_.*/RightPlane", right_plane_cfg, translation=(0.7, -1.5, 0.25))

        # add ground plane
        left_plane_cfg = sim_utils.CuboidCfg(size=(4.0, 0.01, 0.5), 
                                                   visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=background_color))
        left_plane_cfg.func("/World/envs/env_.*/LeftPlane", right_plane_cfg, translation=(0.7, 1.5, 0.25))


        self.targets = []
        diffuse_colors = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 1.0, 0.0), (1.0, 0.0, 1.0)]
        init_y_poses = [0.0, 0.5, -0.5, 1.0, -1.0]
        for i in range(5):
            cube_cfg = RigidObjectCfg(
                prim_path=f"/World/envs/env_.*/Goal_{i}",
                spawn=sim_utils.SphereCfg(radius=0.075,
                                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=diffuse_colors[i]),
                                        rigid_props=sim_utils.RigidBodyPropertiesCfg(
                                            disable_gravity=True,
                                        ),
                                        mass_props=sim_utils.MassPropertiesCfg(mass=1.0)),
                                        init_state=RigidObjectCfg.InitialStateCfg(pos=(2.5, init_y_poses[i], 0.1)))
            
            self.targets.append(RigidObject(cube_cfg))

        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])


        # add articulation to scene
        self.scene.articulations["robot"] = self._robot
        self.scene.sensors["tiled_camera"] = self._tiled_camera
        for i, target in enumerate(self.targets):
            self.scene.rigid_objects[f"target_{i}"] = target
        
        self.dt = self.cfg.sim.dt * self.cfg.decimation

        self.target_idx = torch.randint(low=0,
                                        high=len(self.targets),
                                        size=(self.num_envs,),
                                        device=self.device)
        

        self.csv_path = "/home/nitesh/workspace/offline_rl_test/text2nav/IsaacLab/extras_with_bc.csv"
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
        # If CSV file doesn't exist, write the header
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Success", "Timesteps"])

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # Clamp and scale actions
        self.raw_actions = actions.clone()
        self.actions = torch.clamp(self.raw_actions, -1, 1) * 200 + self._robot.data.joint_pos[:, self._joint_dof_idx]

    def _apply_action(self) -> None:
        obs = self._get_observations()
        _, angle = compute_goal_side(self.robot_position, self._robot.data.root_quat_w, self.goal_cube_position)
        prompts = generate_relative_prompts(angle, self.target_idx, self.INDEX_COLOR, y_thresh=0.5, base="Move towards the ball.")
        embedds = self.embedd_pipe.generate(self.rgb_image, prompts)
        actions = self.rl_model.sample_action(embedds.cpu().numpy())
        actions = torch.tensor(actions).unsqueeze(0).to(device=self.device)
        actions = torch.clamp(actions, -1, 1) * 200 + self._robot.data.joint_pos[:, self._joint_dof_idx]
        self._robot.set_joint_position_target(actions, joint_ids=self._joint_dof_idx)
    
    def _get_observations(self) -> dict:
        self.timesteps += 1
        self.rgb_image = self._tiled_camera.data.output['rgb'].clone()

        self.robot_position = self._robot.data.root_pos_w - self.scene.env_origins
        all_target_positions = torch.stack([
            t.data.root_pos_w for t in self.targets
        ]) 

        env_ids = torch.arange(self.num_envs, device=self.device)

        self.goal_cube_position = all_target_positions[self.target_idx, env_ids]  - self.scene.env_origins

        poses = torch.cat((self.robot_position, self._robot.data.root_quat_w, self.goal_cube_position), dim=1)

        _, angle = compute_goal_side(self.robot_position, self._robot.data.root_quat_w, self.goal_cube_position)

        obs = poses.to(dtype=torch.float32)
        observations = {"policy": obs}
        self.extras = {"rgb":self.rgb_image.to(dtype=torch.uint8), "goal_index":self.target_idx.to(dtype=torch.int16), "angle": angle.to(dtype=torch.float32)}
        # print(f"Goal: {self.INDEX_COLOR[int(self.target_idx[0])]}")
        return observations
    

    def _get_rewards(self) -> torch.Tensor:
        # Calculate distance to goal
        dist_to_goal = torch.norm(self.robot_position - self.goal_cube_position, dim=1)
        
        # Set base reward as a function of distance to goal
        rew = -dist_to_goal * 0.1
        
        # Check if the robot is close enough to the goal for a success condition
        success_condition = dist_to_goal < 0.3
        rew[success_condition] = 10  # High reward for success
        
        # Extract robot's heading quaternion (assuming it's a unit quaternion)
        robot_heading = self._robot.data.root_quat_w
        robot_heading_vector = quaternion_to_direction(robot_heading)

        # Calculate the direction to the goal
        direction_to_goal = self.goal_cube_position - self.robot_position
        direction_to_goal = direction_to_goal / (torch.norm(direction_to_goal, dim=1, keepdim=True))  # Normalize

        # Calculate cosine similarity between robot's heading and the goal direction
        cosine_similarity = torch.sum(robot_heading_vector * direction_to_goal, dim=1)  # Dot product

        # Heading reward: reward for facing the goal
        heading_reward = cosine_similarity * 0.1  # You can adjust the scaling factor as needed
        rew += heading_reward
        
        # Compute done conditions (same logic as _get_dones)
        done = dist_to_goal > self.cfg.max_distance_to_goal
        done = torch.logical_or(done, self.robot_position[:, 0] > self.cfg.max_x)
        done = torch.logical_or(done, self.robot_position[:, 0] < self.cfg.min_x)
        done = torch.logical_or(done, self.robot_position[:, 1] > self.cfg.max_y)
        done = torch.logical_or(done, self.robot_position[:, 1] < self.cfg.min_y)
        
        # Penalty for "done" conditions
        rew[done] = -5
        
        return rew


    def _get_hard_constraints(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, device=self.device)
    

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        # Done if dis=stance to goal is greater than max distance or less than 0.1
        dis = torch.norm(self.robot_position - self.goal_cube_position, dim=1)
        done = torch.logical_or(dis > self.cfg.max_distance_to_goal, dis < 0.3)
        self.success = dis < 0.3
        print(f"Success: {self.success}")
        # done if out of bounds
        done = torch.logical_or(done, self.robot_position[:, 0] > self.cfg.max_x)
        done = torch.logical_or(done, self.robot_position[:, 0] < self.cfg.min_x)
        done = torch.logical_or(done, self.robot_position[:, 1] > self.cfg.max_y)
        done = torch.logical_or(done, self.robot_position[:, 1] < self.cfg.min_y)
        # if both time out and done, set done to true and time out to false
        done = torch.logical_or(done, time_out)
        # if both time out and done, set done to true and time out to false
        # time_out = torch.logical_and(done, time_out)

        return done, time_out


    def _reset_idx(self, env_ids: Sequence[int] | None):
        # Handle default environment indices
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        super()._reset_idx(env_ids)

        joint_pos = self._robot.data.default_joint_pos[env_ids][:, self._joint_dof_idx]

        # Configure root state with environment offsets
        default_root_state = self._robot.data.default_root_state[env_ids].clone()
        default_root_state[:, :3] += self.scene.env_origins[env_ids]
        default_root_state[:, :3] += sample_uniform(
            lower=torch.tensor([-0.05, -0.2, 0.0], device=self.device),
            upper=torch.tensor([0.75, 0.25, 0.0], device=self.device),
            size=(len(env_ids), 3),
            device=self.device,
        )

        euler = sample_uniform(
            lower=torch.tensor([0, 0, -np.pi], device=self.device),
            upper=torch.tensor([0, 0, np.pi], device=self.device),
            size=(len(env_ids), 3),
            device=self.device,
        )
        quat = quat_from_euler_xyz_batch(euler).cpu().numpy()
        default_root_state[:, 3:7] = torch.tensor(quat, device=self.device)

        # Update internal state buffers
        joint_vel = torch.zeros_like(joint_pos, device=self.device)

        # Write states to physics simulation
        self._robot.set_joint_position_target(joint_pos, self._joint_dof_idx, env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, self._joint_dof_idx, env_ids)
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)

        self.target_idx[env_ids] = torch.randint(low=0,
                                                    high=len(self.targets),
                                                    size=(len(env_ids),),
                                                    device=self.device)
        # self.target_idx = torch.ones_like(self.target_idx) * 3

        # Convert to NumPy (no flatten!)
        self.success = self.success.float()
        success_np = self.success.detach().cpu().numpy()
        timestep_np = self.timesteps.cpu().numpy()


        # Loop over each env and write a row per env
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            for i in range(len(env_ids)):
                print(success_np)
                print(timestep_np)
                row = [success_np[i].tolist()] + [timestep_np[i].tolist()]
                print(f"Writing row: {row}")
                writer.writerow(row)

        self.timesteps = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)



def generate_relative_prompts(y, goal_index, INDEX_COLOR, y_thresh=0.2, base="Move toward the ball."):
    """
    goal_vec_robot: (B, 3) tensor, goal vector in robot frame
    y_thresh: float, threshold for deciding clear left/right
    Returns: List of strings (prompts)
    """
    prompts = []

    for i in range(len(y)):
        if y[i] > y_thresh:
            prompts.append(f"The target is {INDEX_COLOR[int(goal_index[i])]} ball which is to your left. {base}")
        elif y[i] < -y_thresh:
            prompts.append(f"The target is {INDEX_COLOR[int(goal_index[i])]} ball which is to your right. {base}")
        else:
            prompts.append(f"The target is {INDEX_COLOR[int(goal_index[i])]} ball which is straight ahead. {base}")
    
    return prompts


def batch_quat_to_rot_matrix(quat):
    """Convert batch of quaternions [w, x, y, z] to rotation matrices (B, 3, 3)."""
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    
    B = quat.shape[0]
    rot_mats = torch.empty((B, 3, 3), dtype=quat.dtype, device=quat.device)

    rot_mats[:, 0, 0] = 1 - 2 * (y**2 + z**2)
    rot_mats[:, 0, 1] = 2 * (x*y - z*w)
    rot_mats[:, 0, 2] = 2 * (x*z + y*w)

    rot_mats[:, 1, 0] = 2 * (x*y + z*w)
    rot_mats[:, 1, 1] = 1 - 2 * (x**2 + z**2)
    rot_mats[:, 1, 2] = 2 * (y*z - x*w)

    rot_mats[:, 2, 0] = 2 * (x*z - y*w)
    rot_mats[:, 2, 1] = 2 * (y*z + x*w)
    rot_mats[:, 2, 2] = 1 - 2 * (x**2 + y**2)

    return rot_mats

@torch.jit.script
def compute_goal_side(robot_pos, robot_quat, goal_pos):
    """
    Returns a tensor of strings: "LEFT" or "RIGHT" for each batch element.
    Shapes:
        robot_pos: (B, 3)
        robot_quat: (B, 4) in [w, x, y, z]
        goal_pos: (B, 3)
    """
    goal_vec_world = goal_pos - robot_pos  # (B, 3)
    rot_mats = batch_quat_to_rot_matrix(robot_quat)  # (B, 3, 3)
    rot_mats_inv = rot_mats.transpose(1, 2)  # Inverse of rotation matrix
    goal_vec_robot = torch.bmm(rot_mats_inv, goal_vec_world.unsqueeze(-1)).squeeze(-1)  # (B, 3)

    # +1 for LEFT, -1 for RIGHT
    side = torch.where(goal_vec_robot[:, 1] > 0, torch.tensor(1), torch.tensor(-1))

    return side, goal_vec_robot[:, 1]


@torch.jit.script
def quat_from_euler_xyz_batch(euler_angles: torch.Tensor) -> torch.Tensor:
    """Convert batched Euler angles (XYZ convention) to quaternions.

    Args:
        euler_angles: Tensor of shape (N, 3) where each row is (roll, pitch, yaw) in radians.

    Returns:
        Tensor of shape (N, 4) representing (w, x, y, z) quaternions.
    """
    roll = euler_angles[:, 0]
    pitch = euler_angles[:, 1]
    yaw = euler_angles[:, 2]

    half_roll = roll * 0.5
    half_pitch = pitch * 0.5
    half_yaw = yaw * 0.5

    cr = torch.cos(half_roll)
    sr = torch.sin(half_roll)
    cp = torch.cos(half_pitch)
    sp = torch.sin(half_pitch)
    cy = torch.cos(half_yaw)
    sy = torch.sin(half_yaw)

    qw = cy * cr * cp + sy * sr * sp
    qx = cy * sr * cp - sy * cr * sp
    qy = cy * cr * sp + sy * sr * cp
    qz = sy * cr * cp - cy * sr * sp

    return torch.stack((qw, qx, qy, qz), dim=1)        

@torch.jit.script
def quaternion_to_direction(quaternions):
    """
    Converts a batch of quaternions to unit vectors representing the forward direction.

    Args:
        quaternions (torch.Tensor): A tensor of shape (n, 4), where each row is a quaternion (w, x, y, z).

    Returns:
        torch.Tensor: A tensor of shape (n, 3), where each row is a unit vector representing the forward direction.
    """
    # Extract quaternion components: w, x, y, z
    w = quaternions[:, 0]
    x = quaternions[:, 1]
    y = quaternions[:, 2]
    z = quaternions[:, 3]
    
    # Compute the forward direction vector for each quaternion
    forward_x = 2 * (x * z - w * y)
    forward_y = 2 * (y * z + w * x)
    forward_z = w**2 - x**2 - y**2 + z**2
    
    # Stack the components to create a tensor of direction vectors
    direction = torch.stack([forward_x, forward_y, forward_z], dim=1)
    
    # Normalize the direction vectors to ensure they are unit vectors
    direction = direction / torch.norm(direction, dim=1, keepdim=True)
    
    return direction


