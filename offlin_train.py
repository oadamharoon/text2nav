import pickle
import numpy as np
from d3rlpy.algos import DQNConfig, SACConfig, IQLConfig, TD3PlusBCConfig, BCConfig, CQLConfig, DiscreteBCConfig
from d3rlpy.preprocessing import ActionScaler, MinMaxActionScaler
with open("/work/mech-ai-scratch/nitesh/workspace/text2nav/embeddings_buffer.pkl", "rb") as f:
    replay_buffer_with_embeddings = pickle.load(f)

# %%
embedds = np.array([item[0] for item in replay_buffer_with_embeddings])
actions = np.array([item[1] for item in replay_buffer_with_embeddings])
rewards = np.array([item[2] for item in replay_buffer_with_embeddings])
dones = np.array([item[3] for item in replay_buffer_with_embeddings])
truncateds = np.array([item[4] for item in replay_buffer_with_embeddings])

# %%
embedds_sliced = []
actions_sliced = []
rewards_sliced = []
dones_sliced = []
truncateds_sliced = []
for i in range(embedds.shape[1]):
    done_mask = dones[:, 0, :] > 0
    truncated_mask = truncateds[:, 0, :] > 0

    # Element-wise OR
    final_mask = np.logical_or(done_mask, truncated_mask)

    # Get the indices where it's True
    indices = np.where(final_mask)[0]
    last_index = indices[-1]+1
    embedds_sliced.append(embedds[:last_index, i, :])
    actions_sliced.append(actions[:last_index, i, :])
    rewards_sliced.append(rewards[:last_index, i, :])
    dones_sliced.append(dones[:last_index, i, :])
    truncateds_sliced.append(truncateds[:last_index, i, :])


embedds_sliced = np.vstack(embedds_sliced)
actions_sliced = np.vstack(actions_sliced)
rewards_sliced = np.vstack(rewards_sliced)
dones_sliced = np.vstack(dones_sliced)
truncateds_sliced = np.vstack(truncateds_sliced)
embedds_sliced.shape, actions_sliced.shape, rewards_sliced.shape, dones_sliced.shape, truncateds_sliced.shape


# %%
import d3rlpy

# %%
from d3rlpy.dataset import MDPDataset

# %%
def create_dataset(embeddings, actions, rewards, dones, truncateds):
    dataset = MDPDataset(
        observations=embeddings,
        actions=actions,
        rewards=rewards,
        terminals=dones,
        timeouts=truncateds,
        action_space=d3rlpy.constants.ActionSpace.CONTINUOUS,
    )
    return dataset

# %%
dataset = create_dataset(
        embeddings=embedds_sliced,
        actions=actions_sliced,
        rewards=rewards_sliced,
        dones=dones_sliced,
        truncateds=truncateds_sliced
    )

encoder_factory = d3rlpy.models.VectorEncoderFactory(
    hidden_units=[1024, 512, 256, 128, 64],
    activation='elu',
)

# %%

bc = TD3PlusBCConfig(batch_size=512, actor_encoder_factory=encoder_factory, critic_encoder_factory=encoder_factory).create(device="cuda:0")
bc.build_with_dataset(dataset)
# bc.load_model("td3_bc_model.pt")

# %%
bc.fit(
    dataset,
    n_steps=1000000,
    save_interval=10,
)

bc.save_model("td3_bc_model_v2.pt")

