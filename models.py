import torch
from torch import nn
import numpy as np
from torch.nn import functional as F

class MeNet5(nn.Module):
    def __init__(
        self, embedding_dimension: int = 64, input_channels: int = 1, width_factor: int = 1
    ):
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.input_channels = input_channels
        self.width_factor = width_factor
        self.layer1 = nn.Sequential(
            nn.Conv2d(
                input_channels, 16 * width_factor, kernel_size=5, stride=2, padding=2
            ),
            nn.ReLU(),
            nn.BatchNorm2d(16 * width_factor),
            nn.Conv2d(
                16 * width_factor, 32 * width_factor, kernel_size=5, stride=2, padding=2
            ),
            nn.ReLU(),
            nn.BatchNorm2d(32 * width_factor),
            nn.Conv2d(
                32 * width_factor, 32 * width_factor, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.BatchNorm2d(32 * width_factor),
            nn.AvgPool2d(2, stride=2),
        )
        self.fc = nn.Linear(9 * 32 * width_factor, embedding_dimension)

    def forward(self, x):
        out = self.layer1(x)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

class RNNPredictor(torch.nn.Module):
    def __init__(
        self, hidden_size: int = 512, num_layers: int = 1, action_dimension: int = 2
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnn = torch.nn.GRU(
            input_size=action_dimension,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
        )

    def predict_sequence(
        self, first_hidden_state: torch.Tensor, actions: torch.Tensor
    ):
        return self.rnn(actions, first_hidden_state.unsqueeze(0).repeat(self.num_layers, 1, 1))[0]
