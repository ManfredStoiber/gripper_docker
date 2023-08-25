
import torch
import torch.nn as nn


class Pixel_Critic(nn.Module):
    def __init__(self, latent_size, num_actions, encoder):
        super(Pixel_Critic, self).__init__()

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

    def forward(self, state, action, goal):
        z_vector   = self.encoder_net(state)
        obs_action = torch.cat([z_vector, action, goal], dim=1)
        q1 = self.Q1(obs_action)
        q2 = self.Q2(obs_action)
        return q1, q2