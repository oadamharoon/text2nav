# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
from typing import Tuple

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
from omni.isaac.lab.utils.math import sample_uniform, euler_xyz_from_quat
from omni.isaac.lab.assets import RigidObject, RigidObjectCfg
from omni.isaac.core.utils.nucleus import get_assets_root_path

import gymnasium as gym
import numpy as np


@configclass
class JetbotCfg(DirectRLEnvCfg):
    # env
    decimation = 1
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
    observation_space = {"rgb": gym.spaces.Box(low=0, high=255, shape=(tiled_camera.height, tiled_camera.width, 3), dtype=np.uint8), 
                         "poses": gym.spaces.Box(low=-20.00, high=20.0, shape=(6,), dtype=np.float32),
                         "goal_index": gym.spaces.Discrete(5)}
    
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

        self.translations_tensor = torch.tensor(np.array(self.translations)).unsqueeze(0).repeat(self.num_envs, 1, 1)
        print(self.translations_tensor)
        print(self.scene.env_origins)


    def close(self):
        """Cleanup for the environment."""
        super().close()

    def _setup_scene(self):
        self.stage = stage_utils.get_current_stage()
        self._robot = Articulation(self.cfg.robot_cfg)
        self._tiled_camera = TiledCamera(self.cfg.tiled_camera)
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg(color=(0.67, 0.84, 0.9)))

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
        props = [
            ("TrashCan", "SM_TrashCan_02.usd"),
            ("WaterDispenser", "SM_WaterDispenser_01a.usd"),
            ("Table", "SM_SideTable_02a.usd")]
        # Setup complex hospital env
        init_y_poses = [0.0, 0.5, -0.5, 1.0, -1.0]


        self.translations = []
        for i, (name, usd_file) in enumerate(props):
            prop_cfg = sim_utils.UsdFileCfg(
                usd_path=get_assets_root_path() + f"/Isaac/Environments/Hospital/Props/{usd_file}",
                scale=(0.25, 0.25, 0.25)  # Adjust per prop if needed
            )

            # Replace `func` with however your sim handles spawning UsdFileCfgs
            prop_cfg.func(
                f"/World/envs/env_.*/{name}",
                prop_cfg,
                translation=(2.5, init_y_poses[i], 0.1)
            )
            self.translations.append([2.5, init_y_poses[i], 0.1])

            # self.targets.append(prop_cfg)


        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])


        # add articulation to scene
        self.scene.articulations["robot"] = self._robot
        self.scene.sensors["tiled_camera"] = self._tiled_camera
        for i, target in enumerate(self.targets):
            self.scene.rigid_objects[f"target_{i}"] = target
        
        self.dt = self.cfg.sim.dt * self.cfg.decimation

        self.target_idx = torch.randint(low=0,
                                        high=len(self.translations),
                                        size=(self.num_envs,),
                                        device=self.device)




    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # Clamp and scale actions
        self.raw_actions = actions.clone()
        # print(f"Raw actions: {self.raw_actions}")
        self.actions = torch.clamp(self.raw_actions, -1, 1) * 500 + self._robot.data.joint_pos[:, self._joint_dof_idx]

    def _apply_action(self) -> None:
        self._robot.set_joint_position_target(self.actions, joint_ids=self._joint_dof_idx)
    
    def _get_observations(self) -> dict:
        self.rgb_image = self._tiled_camera.data.output['rgb'].clone()

        self.robot_position = self._robot.data.root_pos_w - self.scene.env_origins
        all_target_positions = self.translations_tensor
        # all_target_positions = torch.stack([
        #     t.data.root_pos_w for t in self.targets
        # ]) 

        env_ids = torch.arange(self.num_envs, device=self.device)

        self.goal_cube_position = all_target_positions[self.target_idx, env_ids]  - self.scene.env_origins

        poses = torch.cat((self.robot_position, self.goal_cube_position), dim=1)

        # Observations
        obs = {"rgb":self.rgb_image.to(dtype=torch.float32), 'poses':poses.to(dtype=torch.float32), "goal_index":self.target_idx.to(dtype=torch.int16)}
        observations = {"policy": obs}
        return observations
    

    # def _get_rewards(self) -> torch.Tensor:
    #     # Calculate distance to goal
    #     dist_to_goal = torch.norm(self.robot_position - self.goal_cube_position, dim=1)
    #     robot_heading = self._robot.data.body_com_quat_w
    #     rew = -dist_to_goal * 0.1
    #     # Compute done conditions (same logic as _get_dones)
    #     done = torch.logical_or(dist_to_goal > self.cfg.max_distance_to_goal, dist_to_goal < 0.1)
    #     done = torch.logical_or(done, self.robot_position[:, 0] > self.cfg.max_x)
    #     done = torch.logical_or(done, self.robot_position[:, 0] < self.cfg.min_x)
    #     done = torch.logical_or(done, self.robot_position[:, 1] > self.cfg.max_y)
    #     done = torch.logical_or(done, self.robot_position[:, 1] < self.cfg.min_y)
    #     rew[done] = -5
    #     return rew

    def _get_rewards(self) -> torch.Tensor:
        # Vector from robot to goal
        to_goal_vec = self.goal_cube_position - self.robot_position
        to_goal_dir = torch.nn.functional.normalize(to_goal_vec, dim=1)

        # Assume robot_heading is a normalized 2D heading vector (e.g., [cos(theta), sin(theta)])
        # If it's a quaternion, you'd need to extract the heading direction
        # print(self._robot.data.body_com_quat_w)
        robot_heading = get_robot_heading_vector(self._robot.data.root_com_quat_w)  # Implement this if not already available

        # Compute heading alignment: dot product is 1 if perfectly aligned, -1 if opposite
        heading_alignment = torch.sum(robot_heading * to_goal_dir[:, :2], dim=1)

        # Calculate distance to goal
        dist_to_goal = torch.norm(to_goal_vec, dim=1)

        # Distance-based reward
        dist_rew = -dist_to_goal * 0.1

        # Heading bonus: scaled by alignment (optional weight)
        heading_bonus = 0.1 * heading_alignment  # Tune weight as needed

        rew = dist_rew + heading_bonus

        # Done conditions
        done = torch.logical_or(dist_to_goal > self.cfg.max_distance_to_goal, dist_to_goal < 0.1)
        done = torch.logical_or(done, self.robot_position[:, 0] > self.cfg.max_x)
        done = torch.logical_or(done, self.robot_position[:, 0] < self.cfg.min_x)
        done = torch.logical_or(done, self.robot_position[:, 1] > self.cfg.max_y)
        done = torch.logical_or(done, self.robot_position[:, 1] < self.cfg.min_y)

        # Apply penalty on done
        rew[done] = -5

        return rew
    


    def _get_hard_constraints(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, device=self.device)
    

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        # Done if dis=stance to goal is greater than max distance or less than 0.1
        dis = torch.norm(self.robot_position - self.goal_cube_position, dim=1)
        done = torch.logical_or(dis > self.cfg.max_distance_to_goal, dis < 0.1)
        # done if out of bounds
        done = torch.logical_or(done, self.robot_position[:, 0] > self.cfg.max_x)
        done = torch.logical_or(done, self.robot_position[:, 0] < self.cfg.min_x)
        done = torch.logical_or(done, self.robot_position[:, 1] > self.cfg.max_y)
        done = torch.logical_or(done, self.robot_position[:, 1] < self.cfg.min_y)
        # if both time out and done, set done to true and time out to false
        done = torch.logical_or(done, time_out)
        # if both time out and done, set done to true and time out to false
        time_out = torch.logical_and(done, time_out)

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
        

@torch.jit.script
def get_robot_heading_vector(quat) -> torch.Tensor:
    """
    Converts robot orientation quaternions to 2D heading vectors (XY plane).
    Assumes robot faces forward along the local +X axis.
    """
    # quat = self._robot.data.body_com_quat_w  # shape (N, 4), assumed format: [w, x, y, z]
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

    # Compute the robot's forward direction in world space using quaternion rotation
    # Assuming forward vector in local frame is (1, 0, 0)
    # Rotated vector: v' = q * v * q_conj

    # Quaternion multiplication for rotating (1, 0, 0) direction
    # Simplified for local forward [1, 0, 0], the resulting world direction is:
    forward_x = 1 - 2 * (y ** 2 + z ** 2)
    forward_y = 2 * (x * y + w * z)

    heading_vec = torch.stack([forward_x, forward_y], dim=1)
    heading_vec = torch.nn.functional.normalize(heading_vec, dim=1)

    return heading_vec  # shape: (N, 2)
