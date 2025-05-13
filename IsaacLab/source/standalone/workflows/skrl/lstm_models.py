import gymnasium as gym

import numpy as np
import torch
import torch.nn as nn
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.utils import set_seed


class Policy(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum",
                 num_envs=1, num_layers=1, hidden_size=64, sequence_length=20):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)

        self.num_envs = num_envs
        self.num_layers = num_layers
        self.hidden_size = hidden_size  # Hcell (Hout is Hcell because proj_size = 0)
        self.sequence_length = sequence_length
        self.num_actions = action_space.shape[0]

        # Define separate feature extractors for different observation types
        self.image_features_container = nn.Sequential(
            nn.LazyConv2d(out_channels=32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.LazyConv2d(out_channels=64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.LazyConv2d(out_channels=64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.ee_features_container = nn.Sequential(
            nn.LazyLinear(out_features=64),
            nn.ReLU(),
            nn.LazyLinear(out_features=64),
            nn.ReLU(),
        )
        self.joint_features_container = nn.Sequential(
            nn.LazyLinear(out_features=64),
            nn.ReLU(),
            nn.LazyLinear(out_features=64),
            nn.ReLU(),
        )

        # LSTM for processing sequential data
        self.lstm = nn.LSTM(input_size=64 * 3,  # Assuming concatenated features have size 64 * 3
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layers,
                            batch_first=True)  # batch_first -> (batch, sequence, features)

        # Final network to output actions
        self.net = nn.Sequential(nn.Linear(self.hidden_size, 64),
                                 nn.ReLU(),
                                 nn.Linear(64, self.num_actions))
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def get_specification(self):
        # batch size (N) is the number of envs
        return {"rnn": {"sequence_length": self.sequence_length,
                        "sizes": [(self.num_layers, self.num_envs, self.hidden_size),    # hidden states (D ∗ num_layers, N, Hout)
                                  (self.num_layers, self.num_envs, self.hidden_size)]}}  # cell states   (D ∗ num_layers, N, Hcell)

    def compute(self, inputs):
        states = inputs["states"]
        terminated = inputs.get("terminated", None)
        hidden_states, cell_states = inputs["rnn"][0], inputs["rnn"][1]

        # Extract features from different observation types
        image_features = self.image_features_container(torch.permute(states['rgb'], (0, 3, 1, 2)))
        ee_features = self.ee_features_container(states['ee_position'])
        joint_features = self.joint_features_container(states['joints'])

        # Combine features
        combined_features = torch.cat((image_features, ee_features, joint_features), dim=1)

        # Reshape combined features for LSTM input
        combined_features = combined_features.view(-1, self.sequence_length, combined_features.shape[-1])

        # training
        if self.training:
            hidden_states = hidden_states.view(self.num_layers, -1, self.sequence_length, hidden_states.shape[-1])  # (D * num_layers, N, L, Hout)
            cell_states = cell_states.view(self.num_layers, -1, self.sequence_length, cell_states.shape[-1])  # (D * num_layers, N, L, Hcell)
            # get the hidden/cell states corresponding to the initial sequence
            hidden_states = hidden_states[:,:,0,:].contiguous()  # (D * num_layers, N, Hout)
            cell_states = cell_states[:,:,0,:].contiguous()  # (D * num_layers, N, Hcell)

            # reset the RNN state in the middle of a sequence
            if terminated is not None and torch.any(terminated):
                rnn_outputs = []
                terminated = terminated.view(-1, self.sequence_length)
                indexes = [0] + (terminated[:,:-1].any(dim=0).nonzero(as_tuple=True)[0] + 1).tolist() + [self.sequence_length]

                for i in range(len(indexes) - 1):
                    i0, i1 = indexes[i], indexes[i + 1]
                    rnn_output, (hidden_states, cell_states) = self.lstm(combined_features[:,i0:i1,:], (hidden_states, cell_states))
                    hidden_states[:, (terminated[:,i1-1]), :] = 0
                    cell_states[:, (terminated[:,i1-1]), :] = 0
                    rnn_outputs.append(rnn_output)

                rnn_states = (hidden_states, cell_states)
                rnn_output = torch.cat(rnn_outputs, dim=1)
            # no need to reset the RNN state in the sequence
            else:
                rnn_output, rnn_states = self.lstm(combined_features, (hidden_states, cell_states))
        # rollout
        else:
            rnn_output, rnn_states = self.lstm(combined_features, (hidden_states, cell_states))

        # flatten the RNN output
        rnn_output = torch.flatten(rnn_output, start_dim=0, end_dim=1)  # (N, L, D ∗ Hout) -> (N * L, D ∗ Hout)

        # Pendulum-v1 action_space is -2 to 2
        return 2 * torch.tanh(self.net(rnn_output)), self.log_std_parameter, {"rnn": [rnn_states[0], rnn_states[1]]}
    




class Value(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, num_envs=1, num_layers=1, hidden_size=64, sequence_length=20):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions=False)

        self.num_envs = num_envs
        self.num_layers = num_layers
        self.hidden_size = hidden_size  # Hcell (Hout is Hcell because proj_size = 0)
        self.sequence_length = sequence_length
        self.num_actions = action_space.shape[0]

        # Define separate feature extractors for different observation types
        self.image_features_container = nn.Sequential(
            nn.LazyConv2d(out_channels=32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.LazyConv2d(out_channels=64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.LazyConv2d(out_channels=64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.ee_features_container = nn.Sequential(
            nn.LazyLinear(out_features=64),
            nn.ReLU(),
            nn.LazyLinear(out_features=64),
            nn.ReLU(),
        )
        self.joint_features_container = nn.Sequential(
            nn.LazyLinear(out_features=64),
            nn.ReLU(),
            nn.LazyLinear(out_features=64),
            nn.ReLU(),
        )

        # LSTM for processing sequential data
        self.lstm = nn.LSTM(input_size=64 * 3,  # Assuming concatenated features have size 64 * 3
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layers,
                            batch_first=True)  # batch_first -> (batch, sequence, features)

        # Final network to output actions
        self.net = nn.Sequential(nn.Linear(self.hidden_size, 64),
                                 nn.ReLU(),
                                 nn.Linear(64, self.num_actions))
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def get_specification(self):
        # batch size (N) is the number of envs
        return {"rnn": {"sequence_length": self.sequence_length,
                        "sizes": [(self.num_layers, self.num_envs, self.hidden_size),    # hidden states (D ∗ num_layers, N, Hout)
                                  (self.num_layers, self.num_envs, self.hidden_size)]}}  # cell states   (D ∗ num_layers, N, Hcell)

    def compute(self, inputs):
        states = inputs["states"]
        terminated = inputs.get("terminated", None)
        hidden_states, cell_states = inputs["rnn"][0], inputs["rnn"][1]

        # Extract features from different observation types
        image_features = self.image_features_container(torch.permute(states['rgb'], (0, 3, 1, 2)))
        ee_features = self.ee_features_container(states['ee_position'])
        joint_features = self.joint_features_container(states['joints'])

        # Combine features
        combined_features = torch.cat((image_features, ee_features, joint_features), dim=1)

        # Reshape combined features for LSTM input
        combined_features = combined_features.view(-1, self.sequence_length, combined_features.shape[-1])

        # training
        if self.training:
            hidden_states = hidden_states.view(self.num_layers, -1, self.sequence_length, hidden_states.shape[-1])  # (D * num_layers, N, L, Hout)
            cell_states = cell_states.view(self.num_layers, -1, self.sequence_length, cell_states.shape[-1])  # (D * num_layers, N, L, Hcell)
            # get the hidden/cell states corresponding to the initial sequence
            hidden_states = hidden_states[:,:,0,:].contiguous()  # (D * num_layers, N, Hout)
            cell_states = cell_states[:,:,0,:].contiguous()  # (D * num_layers, N, Hcell)

            # reset the RNN state in the middle of a sequence
            if terminated is not None and torch.any(terminated):
                rnn_outputs = []
                terminated = terminated.view(-1, self.sequence_length)
                indexes = [0] + (terminated[:,:-1].any(dim=0).nonzero(as_tuple=True)[0] + 1).tolist() + [self.sequence_length]

                for i in range(len(indexes) - 1):
                    i0, i1 = indexes[i], indexes[i + 1]
                    rnn_output, (hidden_states, cell_states) = self.lstm(combined_features[:,i0:i1,:], (hidden_states, cell_states))
                    hidden_states[:, (terminated[:,i1-1]), :] = 0
                    cell_states[:, (terminated[:,i1-1]), :] = 0
                    rnn_outputs.append(rnn_output)

                rnn_states = (hidden_states, cell_states)
                rnn_output = torch.cat(rnn_outputs, dim=1)
            # no need to reset the RNN state in the sequence
            else:
                rnn_output, rnn_states = self.lstm(combined_features, (hidden_states, cell_states))
        # rollout
        else:
            rnn_output, rnn_states = self.lstm(combined_features, (hidden_states, cell_states))

        # flatten the RNN output
        rnn_output = torch.flatten(rnn_output, start_dim=0, end_dim=1)  # (N, L, D ∗ Hout) -> (N * L, D ∗ Hout)

        # Pendulum-v1 action_space is -2 to 2
        return 2 * torch.tanh(self.net(rnn_output)), self.log_std_parameter, {"rnn": [rnn_states[0], rnn_states[1]]}