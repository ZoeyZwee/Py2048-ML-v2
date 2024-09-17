"""
Plays games, generating experience we later learn from
"""
import math
from dataclasses import dataclass
from time import sleep

import numpy as np

import game2048


class Node:
    def __init__(self, parent, sim, reward):
        self.matrix = None
        self.visits = 0
        self.parent = parent
        self.children = []
        self.is_leaf = True
        self.sim = sim
        self.reward = reward  # transition reward for entering this state (or afterstate)
        self.total_value = 0  # sum of td returns

    def value(self):
        if self.visits == 0:
            return 0

        return self.total_value / self.visits

    def expand(self, network):
        raise NotImplemented

class DecisionNode(Node):
    def __init__(self, parent, sim):
        super().__init__(parent, sim, 0)
        self.is_chance = False
        self.prior_dist = None  # policy is initialized when a decision node is expanded

    def expand(self, network):
        self.is_leaf = False
        self.prior_dist = network.get_policy(self.matrix)
        old_matrix = self.sim.matrix.copy()
        for action in self.sim.action_space:
            reward = self.sim.apply_action(action)
            child = ChanceNode(self, self.sim, reward)
            child.matrix = self.sim.matrix.copy()
            self.children.append(child)
            self.sim.set_matrix(old_matrix)


class ChanceNode(Node):
    def __init__(self, parent, sim, reward):
        super().__init__(parent, sim, reward)
        self.is_chance = True

    def expand(self, network):
        self.is_leaf = False
        old_matrix = self.sim.matrix.copy()
        for code in self.sim.chance_codes:
            self.sim.apply_chance_outcome(code)
            child = DecisionNode(self, self.sim)
            child.matrix = self.sim.matrix.copy()
            self.children.append(child)
            self.sim.set_matrix(old_matrix)

class Actor:
    def __init__(self, config, network):
        self.config = config
        self.network = network
        self.game = game2048.Game()
        self.sim = game2048.Game()
        self.sim.set_matrix(self.game.matrix)
        self.training_stage = 0  # for determining softmax temperature in visits_to_policy()

    def mcts(self):
        """
        Use MCTS to determine which move to take from the root state
        :return: action to take (from game.action_space)
        """
        self.sim.set_matrix(self.game.matrix)
        root = DecisionNode(None, self.sim)
        root.matrix = self.sim.matrix.copy()
        root.expand(self.network)
        self.add_dirichlet_noise_root(root)
        mask_policy(root.prior_dist, self.sim.get_legal_moves())
        # min and max Q values seen so far in search
        Q_min = 0  # leaves have 0 value
        Q_max = self.network.get_value(self.sim.matrix)
        for _ in range(self.config.MCTS_iterations):
            node = root
            # selection
            while not node.is_leaf:
                if node.is_chance:
                    outcome = self.sim.generate_chance_outcome()
                    self.sim.apply_chance_outcome(outcome)
                    node = node.children[outcome]
                else:
                    action = max(self.sim.action_space,
                                 key=lambda a: self.PUCT(node, node.children[a], node.prior_dist[a], Q_min, Q_max)
                                 )
                    self.sim.apply_action(action)
                    node = node.children[action]

            # expansion
            node.expand(self.network)

            # backprop
            if node.is_chance:
                td_return = self.network.get_action_value(self.sim.matrix, action)
            else:
                td_return = self.network.get_value(self.sim.matrix)

            while node is not None:
                td_return = td_return * self.config.td_discount + node.reward
                node.visits += 1
                node.total_value += td_return
                Q_min = min(node.value(), Q_min)
                Q_max = max(node.value(), Q_max)
                node = node.parent

        visits = np.array([child.visits for child in root.children], dtype=np.float32)
        return root.value(), visits

    def PUCT(self, parent, child, prior, Q_min, Q_max):
        """
        Compute the PUCT score of a child, given its parent.
        :param parent: parent Node
        :param child: child Node we want the PUCT score for
        :param prior: prior probability of choosing this child
        :param Q_min: lowest Q-value in this tree. used for normalizing Q-values
        :param Q_max: highest Q-value in this tree. used for normalizing Q-values
        :return: PUCT score
        """
        exploration_factor = self.config.ucb_c1 + math.log(parent.visits + self.config.ucb_c2 + 1) / self.config.ucb_c2
        Q_norm = (child.value() - Q_min) / (Q_max - Q_min)
        return Q_norm + prior + math.sqrt(parent.visits / (1 + child.visits)) * exploration_factor

    def add_dirichlet_noise_root(self, node):
        alpha = self.config.dirichlet_alpha
        fraction = self.config.dirichlet_fraction
        node.prior_dist *= (1-fraction)
        node.prior_dist += fraction*alpha

    def choose_action_from_visits(self, visits):
        # after many many training stages, just act greedily
        if self.training_stage == 4:
            return visits.argmax()
        else:
            # update temperature based on number of training steps
            for i, threshold in enumerate(self.config.training_stages):
                if self.network.training_steps > threshold:
                    self.training_stage = i

            # apply softmax
            policy = np.exp(visits) / self.config.temperature_stages[self.training_stage]
            policy = policy / policy.sum()
            return np.random.choice([0,1,2,3], p=policy)

def mask_policy(policy, is_legal):
    all_false = all(is_legal)
    for i in range(4):
        if not is_legal[i]:
            policy[i] = float(all_false)  # set to 0, unless all moves are illegal then set 1
    policy /= sum(policy)

@dataclass
class Config:
    ucb_c1: float
    ucb_c2: float
    td_discount: float
    MCTS_iterations: int
    dirichlet_fraction: float
    dirichlet_alpha: float
    temperature_stages: list[float]
    training_stages: list[float]

if __name__ == "__main__":
    from network import Network
    print("starting")
    network = Network()
    config = Config(1.25, 19652, 0.999, 100, 0.1, 0.25, [1.0, 0.5, 0.1], [1e3, 1e4, 1e5])
    actor = Actor(config, network)
    ui = game2048.GameWindow()
    ui.paint(actor.game)

    while not actor.game.is_game_over():
        value, visits = actor.mcts()
        action = actor.choose_action_from_visits(visits)

        reward = actor.game.apply_action(action)
        ui.paint(actor.game)
        outcome = actor.game.generate_chance_outcome()
        actor.game.apply_chance_outcome(outcome)
        ui.paint(actor.game)

