import gymnasium as gym
from gym import spaces
import numpy as np
from typing import Any, Dict, Tuple
from robot import HelloWorld as jetbot


class JetbotEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float16)  # Example action space
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(256, 256, 3), dtype=np.uint8)  # Example observation space
        self.state = None
        self.done = False
        self.jetbot = jetbot()
        self.jetbot.setup_scene()
        self.world = self.jetbot.world
        self.world.step()
    

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        # Apply action to the robot
        self.jetbot.send_robot_actions(action)
        
        # Step the simulation
        self.world.step()
        
        # Get observation
        observation = self.jetbot.get_observations()
        
        # Calculate reward
        reward = self.jetbot.distance2goal()
        
        # Check if done
        self.done = self.check_done()
        
        return observation, reward, self.done, False, {}
    
    def check_done(self, reward) -> bool:
        if reward < 0.1:
            return True
        return False

    def reset(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        # Reset the environment
        self.world.reset()
        self.done = False
        
        # Get initial observation
        observation = self.jetbot.get_observations()
        
        return observation, {}