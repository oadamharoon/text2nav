import d3rlpy
import torch


device = "cuda" if torch.cuda.is_available() else "cpu"
# setup algorithm
random_policy = d3rlpy.algos.SACConfig().create(device=device)
# load dataset
dataset = d3rlpy.dataset.load_v1("text2nav/replay_buffer_converted.h5")
print(dataset)