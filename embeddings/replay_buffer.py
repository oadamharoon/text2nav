import torch
import random
import numpy as np
from collections import deque

class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)

    def store(self, state, action, reward, next_state, done):
        """ Store an experience tuple in the replay buffer """
        self.buffer.append((state, action, reward, next_state, done))

    def __len__(self):
        return len(self.buffer)
