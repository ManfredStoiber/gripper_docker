
import torch
import torch.nn as nn
from networks.weight_bias_init import weight_init


class AE_Critic(nn.Module):
    def __init__(self, latent_size, num_actions, encoder):
        super(AE_Critic, self).__init__()

        self.encoder_net = encoder
        self.hidden_size = [1024, 1024]

        self.Q1 = nn.Sequential(
            nn.Linear(latent_size + num_actions, self.hidden_size[0]),
            nn.ReLU(),
            nn.BatchNorm1d(self.hidden_size[0]),
            nn.Linear(self.hidden_size[0], self.hidden_size[1]),
            nn.ReLU(),
            nn.Linear(self.hidden_size[1], 1)
        )

        self.Q2 = nn.Sequential(
            nn.Linear(latent_size + num_actions, self.hidden_size[0]),
            nn.ReLU(),
            nn.BatchNorm1d(self.hidden_size[0]),
            nn.Linear(self.hidden_size[0], self.hidden_size[1]),
            nn.ReLU(),
            nn.Linear(self.hidden_size[1], 1)
        )

        self.apply(weight_init)

    def forward(self, state, action, goal, detach_encoder=False):
        z_vector   = self.encoder_net(state, detach=detach_encoder)
        obs_action = torch.cat([z_vector, action, goal], dim=1)
        q1 = self.Q1(obs_action)
        q2 = self.Q2(obs_action)
        return q1, q2
