# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to play a checkpoint of an RL agent from skrl.

Visit the skrl documentation (https://skrl.readthedocs.io) to see the examples structured in
a more user-friendly way.
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from skrl.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument(
    "--ml_framework",
    type=str,
    default="torch",
    choices=["torch", "jax", "jax-numpy"],
    help="The ML framework used for training the skrl agent.",
)
parser.add_argument(
    "--algorithm",
    type=str,
    default="PPO",
    choices=["PPO", "IPPO", "MAPPO"],
    help="The RL algorithm used for training the skrl agent.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch

import skrl
from packaging import version

# check for minimum supported skrl version
SKRL_VERSION = "1.3.0"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(
        f"Unsupported skrl version: {skrl.__version__}. "
        f"Install supported version using 'pip install skrl>={SKRL_VERSION}'"
    )
    exit()

if args_cli.ml_framework.startswith("torch"):
    from skrl.utils.runner.torch import Runner
elif args_cli.ml_framework.startswith("jax"):
    from skrl.utils.runner.jax import Runner

from omni.isaac.lab.envs import DirectMARLEnv, multi_agent_to_single_agent
from omni.isaac.lab.utils.dict import print_dict

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils import get_checkpoint_path, load_cfg_from_registry, parse_env_cfg
from omni.isaac.lab_tasks.utils.wrappers.skrl import SkrlVecEnvWrapper

# import ros
from std_msgs.msg import Float64MultiArray
from rclpy.node import Node
from custom_messages.msg import Obs
import numpy as np
import rclpy
from cv_bridge import CvBridge
import cv2

# config shortcuts
algorithm = args_cli.algorithm.lower()


# def main():
"""Play with skrl agent."""
# configure the ML framework into the global skrl variable
if args_cli.ml_framework.startswith("jax"):
    skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

# parse configuration
env_cfg = parse_env_cfg(
    args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
)
try:
    experiment_cfg = load_cfg_from_registry(args_cli.task, f"skrl_{algorithm}_cfg_entry_point")
except ValueError:
    experiment_cfg = load_cfg_from_registry(args_cli.task, "skrl_cfg_entry_point")

# specify directory for logging experiments (load checkpoint)
log_root_path = os.path.join("logs", "skrl", experiment_cfg["agent"]["experiment"]["directory"])
log_root_path = os.path.abspath(log_root_path)
print(f"[INFO] Loading experiment from directory: {log_root_path}")
# get checkpoint path
if args_cli.checkpoint:
    resume_path = os.path.abspath(args_cli.checkpoint)
else:
    resume_path = get_checkpoint_path(
        log_root_path, run_dir=f".*_{algorithm}_{args_cli.ml_framework}", other_dirs=["checkpoints"]
    )
log_dir = os.path.dirname(os.path.dirname(resume_path))

# create isaac environment
env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

# convert to single-agent instance if required by the RL algorithm
if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
    env = multi_agent_to_single_agent(env)

# wrap for video recording
if args_cli.video:
    video_kwargs = {
        "video_folder": os.path.join(log_dir, "videos", "play"),
        "step_trigger": lambda step: step == 0,
        "video_length": args_cli.video_length,
        "disable_logger": True,
    }
    print("[INFO] Recording videos during training.")
    print_dict(video_kwargs, nesting=4)
    env = gym.wrappers.RecordVideo(env, **video_kwargs)

# wrap around environment for skrl
env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)  # same as: `wrap_env(env, wrapper="auto")`

# configure and instantiate the skrl runner
# https://skrl.readthedocs.io/en/latest/api/utils/runner.html
experiment_cfg["trainer"]["close_environment_at_exit"] = False
experiment_cfg["agent"]["experiment"]["write_interval"] = 0  # don't log to TensorBoard
experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0  # don't generate checkpoints
runner = Runner(env, experiment_cfg)

print(f"[INFO] Loading model checkpoint from: {resume_path}")
runner.agent.load(resume_path)
# set agent to evaluation mode
runner.agent.set_running_mode("eval")
dt = env.unwrapped.physics_dt


class ActionPublisher(Node):
    def __init__(self):
        super().__init__('action_publisher')
        self.publisher = self.create_publisher(Float64MultiArray, 'angles', 10)
        self.cv_bridge = CvBridge()
        self.create_subscription(Obs, 'observations', self.update_observations, 10)
    
    def update_observations(self, data: Obs):
        image = data.image
        image = self.cv_bridge.compressed_imgmsg_to_cv2(image)
        image = cv2.flip(image, 0)
        image = cv2.flip(image, 1)
        cv2.imwrite("/home/nitesh/IsaacLab/image.jpg", image)
        image = image / 255.0
        mean_image = np.mean(image, axis=(0, 1))
        image = image - mean_image
        image = cv2.resize(image, (224, 224))
        
        # Add one channel to the image
        image = np.concatenate([image, np.zeros((image.shape[0], image.shape[1], 1))], axis=2)
        position = np.array([data.position.x, data.position.y, data.position.z])
        observations = np.concatenate([position, np.zeros(4), image.flatten()])
        observations = torch.tensor(observations, dtype=torch.float32, device="cuda:0").unsqueeze(0)
        # print(observations.shape)
        with torch.inference_mode():
            action = runner.agent.act(observations, timestep=0, timesteps=0)[0]
        action = action.cpu().numpy().tolist()
        print(action)
        msg = Float64MultiArray()
        msg.data = action[0]
        self.publisher.publish(msg)


# run the main function
rclpy.init()
node = ActionPublisher()
rclpy.spin(node)
# close sim app
simulation_app.close()
