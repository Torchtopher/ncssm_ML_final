from torch import nn
import torch



class DQN(nn.Module):
    def __init__(self, obs_size=42, action_size=7):
        super().__init__()
        print(f"Using obs size {obs_size} and action size {action_size}")
        self.flatten = nn.Flatten()
        self.model = nn.Sequential(
            nn.Linear(obs_size, 512), # 7*6 42 spaces avaiable
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, action_size), # 7 spaces to play
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.model(x) # logits here because the values are straight from the model, not softmaxxed or normalized
        return logits
