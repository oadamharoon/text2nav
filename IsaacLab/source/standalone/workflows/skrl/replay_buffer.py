from collections import deque
import numpy as np


class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)

    def store(self, state, action, reward, next_state, done, truncated):
        """ Store an experience tuple in the replay buffer """
        self.buffer.append((state, action, reward, next_state, done, truncated))

    def __len__(self):
        return len(self.buffer)

    def observations(self):
        """ Retrieve all the observations (states) from the buffer as a numpy array """
        return np.array([experience[0] for experience in self.buffer])

    def actions(self):
        """ Retrieve all the actions from the buffer as a numpy array """
        return np.array([experience[1] for experience in self.buffer])

    def rewards(self):
        """ Retrieve all the rewards from the buffer as a numpy array """
        return np.array([experience[2] for experience in self.buffer])

    def next_states(self):
        """ Retrieve all the next states from the buffer as a numpy array """
        return np.array([experience[3] for experience in self.buffer])

    def dones(self):
        """ Retrieve all the done flags from the buffer as a numpy array """
        return np.array([experience[4] for experience in self.buffer])
