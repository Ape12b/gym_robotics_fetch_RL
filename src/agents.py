import os
import torch
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(25 + 3, 128),
            nn.ReLU(),
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, 4)
        )

    def forward(self, x):
        action = torch.tanh(self.linear_relu_stack(x))
        return action

class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(32, 128),  # input: obs (25) + action (4)
            nn.ReLU(),
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=-1)  # concatenate along last dim
        q_value = self.linear_relu_stack(x)
        return q_value
    
