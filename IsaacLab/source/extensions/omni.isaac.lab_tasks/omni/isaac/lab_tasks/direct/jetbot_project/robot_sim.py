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
from omni.isaac.lab.utils.math import sample_uniform
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

    def close(self):
        """Cleanup for the environment."""
        super().close()

    def _setup_scene(self):
        self.stage = stage_utils.get_current_stage()
        self._robot = Articulation(self.cfg.robot_cfg)
        self._tiled_camera = TiledCamera(self.cfg.tiled_camera)
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg(color=(0.67, 0.84, 0.9)))
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
        diffuse_colors = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 1.0, 0.0), (1.0, 1.0, 1.0)]
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
        all_target_positions = torch.stack([
            t.data.root_pos_w for t in self.targets
        ]) 

        env_ids = torch.arange(self.num_envs, device=self.device)

        self.goal_cube_position = all_target_positions[self.target_idx, env_ids]  - self.scene.env_origins

        poses = torch.cat((self.robot_position, self.goal_cube_position), dim=1)

        # Observations
        obs = {"rgb":self.rgb_image.to(dtype=torch.float32), 'poses':poses.to(dtype=torch.float32), "goal_index":self.target_idx.to(dtype=torch.int16)}
        observations = {"policy": obs}
        return observations
    

    def _get_rewards(self) -> torch.Tensor:
        # Calculate distance to goal
        dist_to_goal = torch.norm(self.robot_position - self.goal_cube_position, dim=1)
        rew = -dist_to_goal * 0.1
        # Compute done conditions (same logic as _get_dones)
        done = torch.logical_or(dist_to_goal > self.cfg.max_distance_to_goal, dist_to_goal < 0.1)
        done = torch.logical_or(done, self.robot_position[:, 0] > self.cfg.max_x)
        done = torch.logical_or(done, self.robot_position[:, 0] < self.cfg.min_x)
        done = torch.logical_or(done, self.robot_position[:, 1] > self.cfg.max_y)
        done = torch.logical_or(done, self.robot_position[:, 1] < self.cfg.min_y)
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

