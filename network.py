from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
import math


def encode_value(value: float):
    """
    Scale a reward, afterstate value, or state value, and encode it as a categorical vector
    :param value: quantity to be scaled and encoded
    :return: Pytorch Tensor representing the scaled value
    """
    cat = torch.zeros((601,))

    value_scaled = math.sqrt(value + 1) - 1 + 0.001 * value
    idx = math.floor(value_scaled)
    b = value_scaled - idx
    a = 1 - b
    cat[idx] = a
    cat[idx + 1] = b
    return cat


def decode_value(value: torch.Tensor):
    """
    Extract the value given from the categorical vector, and then apply the inverse transform
    :param value: two-hot tensor representing a given (scaled) value
    :return: unscaled valued
    """

    # decode value is expectation of prob vector
    y = value.dot(torch.arange(0., 601.))
    eps = 1e-3
    out = ((math.sqrt(4 * eps * (math.fabs(y) + 1. + eps) + 1.) - 1.) / (2. * eps))**2 - 1.
    return out


def encode_state(matrix: np.ndarray):
    """
    Transform a board state into the representation used by the neural network.
    Each position of the board has its value represented by a one-hot vector of length 31
    The single "True" position in the vector describes the value (exponent) of the position,
        with the caveat that a "True" in position 0 indicates 0, not 2^0
    The complete representation of the board is the concatenation of each position vector
    :param matrix: np.ndarray. each entry x represents 2^x, except 0 which represents 0
    :return: concatenated one-hot representation of board
    """

    board = torch.zeros((496,))
    it = matrix.flat
    for x in it:
        board[31 * (it.index - 1) + x] = True
    return board

def inputs(matrix, direction=None):
    action = torch.zeros((4,))
    if direction is not None:
        action[direction] = 1.
    state = encode_state(matrix)
    return torch.cat([state, action])
@dataclass
class NetworkOutput:
    policy: torch.Tensor
    value_target: torch.Tensor
    action_value_target: torch.Tensor


class Network(nn.Module):
    def __init__(self):
        super().__init__()

        self.training_steps = 0

        # hidden layers
        self.layers = nn.Sequential(
            nn.LayerNorm(500),
            nn.ReLU(),
            nn.Linear(500, 256),
            ResBlock(),
            ResBlock(),
            ResBlock(),
            ResBlock(),
            ResBlock(),  # 5
            ResBlock(),
            ResBlock(),
            ResBlock(),
            ResBlock(),
            ResBlock()  # 10
        )

        # network heads
        self.policy_out = nn.Sequential(
            nn.Linear(256, 4),
            nn.Softmax(0)
        )

        self.value_out = nn.Sequential(
            nn.Linear(256, 601),
            nn.Softmax(0)
        )

        self.action_value_out = nn.Sequential(
            nn.Linear(256, 601),
            nn.Softmax(0)
        )

    def get_value(self, matrix):
        """Query the network for the value of a given board state"""
        x = inputs(matrix)
        x = self.layers(x)
        x = self.value_out(x)
        return decode_value(x.detach())

    def get_policy(self, matrix):
        """Query the network for a prior policy, given the current board state"""
        x = inputs(matrix)
        x = self.layers(x)
        x = self.policy_out(x)
        return x.detach()

    def get_action_value(self, matrix, direction):
        x = inputs(matrix, direction)
        x = self.layers(x)
        x = self.action_value_out(x)
        return decode_value(x.detach())

    def forward(self, x):
        """
        Forward pass the network and read all outputs
        :param x: network input. torch.cat(encoded_board_state, encoded_action)
        :return: Tensors for policy, value, and action value
        """
        x = self.layers(x)
        return self.policy_out(x), self.value_out(x), self.action_value_out(x)


class ResBlock(nn.Module):
    """
    Pre-Activation Residual Block a la ResNet v2 (Ba, et al. 2016)
    """

    def __init__(self):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256),
        )

    def forward(self, x):
        y = self.block(x)
        return x + y
