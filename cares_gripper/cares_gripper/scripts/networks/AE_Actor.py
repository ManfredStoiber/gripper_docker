
import torch
import torch.nn as nn
from networks.weight_bias_init import weight_init


class AE_Actor(nn.Module):
    def __init__(self, latent_size, num_actions, encoder):
        super(AE_Actor, self).__init__()

        self.encoder_net = encoder
        self.hidden_size = [1024, 1024]

        self.act_net = nn.Sequential(
            nn.Linear(latent_size, self.hidden_size[0]),
            nn.ReLU(),
            nn.BatchNorm1d(self.hidden_size[0]),
            nn.Linear(self.hidden_size[0], self.hidden_size[1]),
            nn.ReLU(),
            nn.BatchNorm1d(self.hidden_size[1]),
            nn.Linear(self.hidden_size[1], num_actions),
            nn.Tanh()
        )
        self.apply(weight_init)

    def forward(self, state, goal,  detach_encoder=False):
        z_vector      = self.encoder_net(state, detach=detach_encoder)
        z_vector_goal = torch.cat([z_vector, goal], dim=1)
        output = self.act_net(z_vector_goal)
        return output
