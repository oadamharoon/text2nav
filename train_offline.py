import pickle
import numpy as np
from stable_baselines3.common.buffers import ReplayBuffer
import d3rlpy
from d3rlpy.dataset import MDPDataset

with open('replay_buffer_embedd.pkl', 'rb') as f:
    sb3_buffer = pickle.load(f)

print("Replay buffer loaded")


def to_mdp_dataset(replay_buffer: ReplayBuffer) -> MDPDataset:
    observations = replay_buffer.observations["rgb"]
    observations = observations.reshape(-1, *observations.shape[2:])
    actions = replay_buffer.actions.reshape(-1, replay_buffer.actions.shape[-1])
    rewards = replay_buffer.rewards.reshape(-1, 1)
    terminals = replay_buffer.dones.reshape(-1, 1)
    timeouts = replay_buffer.timeouts.reshape(-1, 1)
    timeouts = np.where(terminals, False, timeouts)
    print(f"Observations shape: {observations.shape}")
    print(f"Actions shape: {actions.shape}")
    print(f"Rewards shape: {rewards.shape}")
    print(f"Terminals shape: {terminals.shape}")
    print(f"Timeouts shape: {timeouts.shape}")


    return MDPDataset(observations=observations,
                      actions=actions,
                      rewards=rewards,
                      terminals=terminals,
                      timeouts=timeouts,
                      transition_picker=d3rlpy.dataset.FrameStackTransitionPicker(n_frames=4))

mdp_dataset  = to_mdp_dataset(sb3_buffer)

from d3rlpy.algos import DQNConfig, SACConfig, IQLConfig, TD3PlusBCConfig

sac = TD3PlusBCConfig(batch_size=128).create(device="cuda:0")
# sac = DecisionTransformerConfig().create(device="cuda:0")
sac.build_with_dataset(mdp_dataset)

from d3rlpy.metrics import TDErrorEvaluator, DiscountedSumOfAdvantageEvaluator

# calculate metrics with training dataset
td_error_evaluator = TDErrorEvaluator(episodes=mdp_dataset.episodes)
discounted_sum_of_advantage_evaluator = DiscountedSumOfAdvantageEvaluator(episodes=mdp_dataset.episodes)

sac.fit(
    mdp_dataset,
    n_steps=1000000,
    n_steps_per_epoch=1000,
)

sac.save_model("td3bc_model.pt")