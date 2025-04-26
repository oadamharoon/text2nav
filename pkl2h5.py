import pickle
import h5py

with open('IsaacLab/SAC/logs/sb3/Isaac-Jetbot-Direct-v0/2025-04-25_16-46-03/replay_buffer.pkl', 'rb') as f:
    data = pickle.load(f)

print(type(data))
print(data)
