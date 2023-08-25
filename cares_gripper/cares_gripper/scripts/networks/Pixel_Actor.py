
import torch
import torch.nn as nn

class Pixel_Actor(nn.Module):
    def __init__(self, latent_size, num_actions, encoder):
        super(Pixel_Actor, self).__init__()

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

    def forward(self, state, goal):
        z_vector      = self.encoder_net(state)
        z_vector_goal = torch.cat([z_vector, goal], dim=1)
        output = self.act_net(z_vector_goal)
        return output