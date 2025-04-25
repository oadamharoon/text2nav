from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
import gymnasium as gym
import numpy as np
import yaml
import torch
from evaluate_utils import generate_models
from rclpy.node import Node
import rclpy
from std_msgs.msg import Float64MultiArray
from custom_messages.msg import Obs
from skrl.resources.preprocessors.torch import RunningStandardScaler
from cv_bridge import CvBridge
import cv2
import os


class PPOAgent(Node):
    def __init__(self):
        super().__init__("ppo_agent")
        self.publisher_ = self.create_publisher(Float64MultiArray, "angles", 10)
        self.bridge = CvBridge()

        device = "cuda:0"
        observation_space = gym.spaces.Dict({"joints": gym.spaces.Box(low=-20.00, high=20.0, shape=(6,), dtype=np.float32),
                         "rgb": gym.spaces.Box(low=-1.0, high=1.0, shape=(256, 256, 5), dtype=np.float32), 
                         "ee_position": gym.spaces.Box(low=-20.00, high=20.0, shape=(3,), dtype=np.float32)})

        action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(6,))
        config_dir = "/home/nitesh/IsaacLab/logs/skrl/mybuddy_direct/with_realsense_camera"
        # Generate the models
        with open(config_dir+"/params/agent.yaml", 'r') as file:
            cfg = yaml.full_load(file)
            if not isinstance(cfg, dict):
                raise ValueError("Configuration file must contain a dictionary")
        models = generate_models(cfg, observation_space, action_space)

        # Define the agent configuration
        agent_cfg = PPO_DEFAULT_CONFIG.copy()
        # agent_cfg['state_preprocessor'] = RunningStandardScaler
        # agent_cfg["state_preprocessor_kwargs"] = {"size": observation_space, "device": device}
        # agent_cfg["value_preprocessor"] = RunningStandardScaler
        # agent_cfg["value_preprocessor_kwargs"] = {"size": 1, "device": device}

        # Instantiate the agent
        self.agent = PPO(models=models['agent'],  # models dict
                    cfg=agent_cfg,  # configuration dict (preprocessors, learning rate schedulers, etc.)
                    observation_space=observation_space,  # observation space
                    action_space=action_space,  # action space
                    device="cuda:0")  # device

        # Load the checkpoint
        self.agent.load(config_dir+"/checkpoints/best_agent.pt")
        self.subscription_ = self.create_subscription(Obs, "observations", self.callback, 10)
        self.subscription_
    
    def _preprocess_obs(self, obs: dict[str, torch.tensor]) -> torch.Tensor:
        # Sort obs keys
        obs = {k: obs[k] for k in sorted(obs.keys())}
        return torch.cat([obs[k].flatten() for k in obs.keys()], dim=0).reshape(1, -1)


    def callback(self, msg: Obs):
        print("Received message")
        rgb = self.bridge.compressed_imgmsg_to_cv2(msg.image) / 255.0
        # Resize and flip the RGB image
        rgb = cv2.resize(rgb, (256, 256))  # Shape: (224, 224, 3)
        rgb = cv2.flip(rgb, 0)  # Flip vertically
        rgb = cv2.flip(rgb, 1)  # Flip horizontally

        # Save the processed image for debugging
        cv2.imwrite("/home/nitesh/IsaacLab/image.jpg", rgb * 255)

        # Normalize the RGB image
        mean_rgb = np.mean(rgb, axis=(0, 1))  # Mean along height and width
        norm_rgb = rgb - mean_rgb  # Normalize by subtracting the mean

        # Create cube_mask and depth_mask
        cube_mask = np.zeros((256, 256, 1), dtype=np.float32)  # Shape: (224, 224, 1)
        depth_mask = np.zeros_like(cube_mask)  # Shape: (224, 224, 1)

        # Concatenate along the last axis (channels)
        image_tensor = np.concatenate((norm_rgb, cube_mask, depth_mask), axis=2)  # Shape: (224, 224, 5)
        joint_tensor = np.array(msg.angles, dtype=np.float32)
        # joint_tensor = np.zeros_like(joint_tensor)
        ee_position_tensor = np.array([msg.position.x, msg.position.y, msg.position.z - 0.15])
        obs = {
            "joints": torch.tensor(joint_tensor, dtype=torch.float32, device="cuda:0"),
            "rgb": torch.tensor(image_tensor, dtype=torch.float32, device="cuda:0"),
            "ee_position": torch.tensor(ee_position_tensor, dtype=torch.float32, device="cuda:0")
        }
        obs = self._preprocess_obs(obs)
        actions = self.agent.act(obs, timestep=0, timesteps=0)[0].detach().cpu().numpy()[0]
        actions_msg = Float64MultiArray()
        actions_msg.data = actions.tolist()
        print(actions_msg.data)
        self.publisher_.publish(actions_msg)

def main(args=None):
    rclpy.init(args=args)
    ppo_agent = PPOAgent()
    rclpy.spin(ppo_agent)
    rclpy.shutdown()


if __name__ == "__main__":
    main()