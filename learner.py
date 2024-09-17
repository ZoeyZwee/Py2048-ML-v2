import numpy as np
import torch

from network import inputs, encode_value


def n_step_return(sample, td_lambda, td_discount):
    # sample is list [NamedTuple(state, visits, action, value, reward)]
    reward_sum = 0     # sum of first k rewards
    lambda_return = 0  # sum of k-step returns
    lambda_factor = 1  # lambda^k
    for step in sample:
        reward_sum *= td_discount
        reward_sum += step.reward
        current_return = reward_sum + td_discount * step.value
        lambda_return += (1-td_lambda) * lambda_factor * current_return
        lambda_factor *= td_lambda
    return lambda_return



class Learner:
    def __init__(self, network):
        self.learning_rate = 3e-4
        self.td_lambda = 0.5
        self.td_discount = 0.999
        self.network = network
        self.adam = torch.optim.Adam(self.network.parameters())

    def training_step(self, samples):
        total_loss = torch.tensor([0.])
        for sample in samples:
            # get target for policy, value, action_value
            value_target = n_step_return(sample, self.td_lambda, self.td_discount)
            value_target = encode_value(value_target)
            sample = sample[0]  # visits, action, state all in first entry in sample
            policy_target = np.exp(sample.visits) / np.sum(np.exp(sample.visits))

            # get estimates for policy, value, action_value
            policy_out, value_out, _ = self.network(inputs(sample.state))
            _, _, action_value_out = self.network(inputs(sample.state, sample.action))

            # all losses are cross-entropy loss. -= so losses are positive (minimize loss)
            total_loss -= torch.dot(torch.tensor(policy_target), torch.log(policy_out))
            total_loss -= torch.dot(value_target, torch.log(value_out))
            total_loss -= torch.dot(value_target, torch.log(action_value_out))


        # compute gradients (presumably they get stored inside the network??)
        total_loss = total_loss / len(samples)
        print(total_loss)
        total_loss.backward()
        # update weights according to gradients (remember self.adam is already linked to the network params)
        self.adam.step()
        self.adam.zero_grad()

        self.network.training_steps += 1

