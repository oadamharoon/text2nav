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

# config shortcuts
algorithm = args_cli.algorithm.lower()

from embeddings import EmbeddingPipeline
embedd_pipeline = EmbeddingPipeline()

def main():
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
    # reset environment
    obs, infos = env.reset()
    timestep = 0
    import time
    start_time = time.time()
    # from d3rlpy.algos import IQLConfig, TD3PlusBCConfig, SACConfig, BCConfig, CQLConfig
    import numpy as np
    import pickle
    import sys
    # import d3rlpy
    # from d3rlpy.preprocessing import ActionScaler, MinMaxActionScaler

    # encoder_factory = d3rlpy.models.VectorEncoderFactory(
    #                         hidden_units=[2000, 2000],
    #                         activation='relu',
    #                     )
    
    # iql = CQLConfig(actor_encoder_factory=encoder_factory, critic_encoder_factory=encoder_factory).create(device="cuda:0")
    # iql.create_impl(observation_shape=(1153,), action_size=2)
    # model_data = torch.load("/home/nitesh/workspace/offline_rl_test/text2nav/cql_model_main_v2.pt")
    # iql._impl.policy.load_state_dict(model_data['policy'])
    replay_buffer = []
    reward = np.zeros((1, 1), dtype=np.float32)
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            # Assuming `infos` is a dictionary where the values are PyTorch tensors on the GPU
            infos = {key: value.cpu() if isinstance(value, torch.Tensor) else value for key, value in infos.items()}

            current_obs = infos
            rgb = current_obs["rgb"].cpu().numpy()
            goal_index = current_obs["goal_index"].cpu()
            angle = current_obs["angle"].cpu()
            actions = runner.agent.act(obs, timestep=0, timesteps=0)[0]
            # print(f"Goal index: {goal_index}")
            # # env stepping
            # _, embedd_obs = embedd_pipeline.generate(rgb, goal_index)
            # # # print(f"Embeddings: {embedd_obs}")
            # actions = iql.sample_action(np.concatenate([np.array(embedd_obs), reward.reshape(-1, 1)], axis=1))
            # # print(f"Actions: {actions}")
            # actions = torch.tensor(actions).unsqueeze(0).to(env_cfg.device)
            obs, reward, done, truncated, infos = env.step(actions)
            reward = reward.cpu().numpy()
            # print("Done: ", done)
            # Assuming `infos` is a dictionary where the values are PyTorch tensors on the GPU
            infos = {key: value.cpu() if isinstance(value, torch.Tensor) else value for key, value in infos.items()}

            # store experience in replay buffer
            # print(f"Done: {done}, \n Truncated: {truncated}")
            replay_buffer.append((rgb, goal_index, angle, actions.cpu(), reward, done.cpu().numpy().astype(int), truncated.cpu().numpy().astype(int)))

            print(f"Replay buffer size: {len(replay_buffer)}")
            if len(replay_buffer) == 5000:
                with open(f"/home/nitesh/workspace/offline_rl_test/text2nav/IsaacLab/replay_buffer.pkl", "wb") as f:
                    pickle.dump(replay_buffer, f)
                sys.exit(0)
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break
        
        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if True and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
