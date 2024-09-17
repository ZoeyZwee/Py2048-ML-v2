import pickle
import random
from dataclasses import dataclass

import torch

from learner import Learner
from network import Network

import game2048
from actor import Actor, Config
import numpy as np


@dataclass
class TrainingExample:
    state: np.ndarray
    visits: np.ndarray
    action: int
    value: float
    reward: float

class ReplayBuffer:
    def __init__(self):
        self.buffer = []  # trajectories of length 200
        self.buffer_size = 125000

    def sample(self):
        """uniformly select a trajectory, then uniformly select a sequence of 10 steps from that trajectory"""
        idx = random.randint(0, len(self.buffer)-1)
        section = self.buffer[idx]
        idx = random.randint(0, len(section)-10)
        return section[idx: idx+10]

    def save_trajectory(self, trajectory):
        if len(trajectory) < 10:
            return
        if len(self.buffer) > self.buffer_size:
            self.buffer = self.buffer[1000:]  # drop 1000 oldest sequences
        self.buffer.append(trajectory)

    def __len__(self):
        return len(self.buffer)

if __name__ == "__main__":
    print("starting")
    network = Network()
    batch_size = 1024
    config = Config(1.25, 19652, 0.999, 100, 0.1, 0.25, [1, 0.5, 0.1], [1e2, 1e4, 1e5])
    actor = Actor(config, network)
    learner = Learner(network)
    ui = game2048.GameWindow()
    buffer = ReplayBuffer()
    render = False
    for i in range(int(1_000_000)):
        print(i)
        trajectory = []
        # play game
        render = (i % 10) == 0
        if render:
            ui.paint(actor.game)
        while not actor.game.is_game_over():

            state = actor.game.matrix.copy()
            value, visits = actor.mcts()
            action = actor.choose_action_from_visits(visits)
            reward = actor.game.apply_action(action)
            # print(game2048.Direction(action).name)
            outcome = actor.game.generate_chance_outcome()
            actor.game.apply_chance_outcome(outcome)
            if render:
                ui.paint(actor.game)
            trajectory.append(TrainingExample(state, visits, action, value, reward))

            if len(trajectory) >= 200:
                print("200 moves played, saving trajectory")
                buffer.save_trajectory(trajectory)
                trajectory = []

                if len(buffer) > 3:
                    print(f"training epoch: {network.training_steps}")
                    batch = [buffer.sample() for _ in range(batch_size)]
                    learner.training_step(batch)

        print("game complete, saving trajectory")
        trajectory[-1].value = 0  # ground the value of terminal states
        buffer.save_trajectory(trajectory)
        trajectory = []
        print(f"max tile achieved:{2**np.max(actor.game.matrix)}")
        actor.game.new_game()

        if len(buffer) > 3:
            print(f"training epoch: {network.training_steps}")
            batch = [buffer.sample() for _ in range(batch_size)]
            learner.training_step(batch)

        if network.training_steps % 100 == 0:
            print("saving network and replay buffer")
            with open("network.pickle", "wb") as f:
                torch.save(network, f)
            with open("replay_buffer.pickle", "wb") as f:
                pickle.dump(buffer, f)
