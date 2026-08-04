import random

import torch.nn as nn


def sample_architecture_config(rng: random.Random) -> dict:
    n_conv_layers = rng.choice([2, 3, 4])
    base_channels = rng.choice([8, 16, 24, 32])
    channels = [base_channels * (2 ** i) for i in range(n_conv_layers)]
    fc_hidden = rng.choice([64, 128, 256])
    return {
        "n_conv_layers": n_conv_layers,
        "channels": channels,
        "fc_hidden": fc_hidden,
    }


class SmallCNN(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        channels = config["channels"]
        in_ch = 1
        conv_blocks = []
        spatial = 28
        for out_ch in channels:
            conv_blocks.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1))
            conv_blocks.append(nn.ReLU(inplace=True))
            conv_blocks.append(nn.MaxPool2d(2))
            in_ch = out_ch
            spatial //= 2
        self.conv = nn.Sequential(*conv_blocks)
        self.fc1 = nn.Linear(in_ch * spatial * spatial, config["fc_hidden"])
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(config["fc_hidden"], 10)

    def forward(self, x):
        x = self.conv(x)
        x = x.flatten(1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)


def build_model(config: dict) -> nn.Module:
    return SmallCNN(config)
