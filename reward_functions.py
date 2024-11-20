import numpy as np
import torch

def return_from_reward(rewards, gamma):
    """
    Compute the discounted returns for each timestep from a tensor of rewards.

    Parameters:
    - rewards (torch.Tensor): Tensor containing the instantaneous rewards.
    - gamma (float): Discount factor (0 < gamma <= 1).

    Returns:
    - torch.Tensor: Tensor containing the discounted returns.
    """
    # Initialize an empty tensor to store the returns
    returns = torch.zeros_like(rewards)

    # Variable to store the accumulated return, initialized to 0
    G = 0

    # Iterate through the rewards in reverse (from future to past)
    for t in reversed(range(len(rewards))):
        # Update the return: G_t = r_t + gamma * G_{t+1}
        G = rewards[t] + gamma * G
        returns[t] = G

    return returns


class ActorCritic(torch.nn.Module):

    def __init__(self, gamma, alpha, stat_gamma=.9, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gamma = gamma
        self.alpha = alpha
        self.debug = False
        self.mean = 0.
        self.std = 1.
        self._stat_gamma = stat_gamma
        self.count = 0.
        self.__name__ = "ActorCritic"

    def forward(self, rewards, value_estimates, log_probs, entropies, to_include=None, is_random=None):
        if self.debug:
            entropies.register_hook(lambda grad: print("Grad H ", torch.abs(grad).sum()))
            log_probs.register_hook(lambda grad: print("Grad LogProb ", torch.abs(grad).sum()))
        returns = return_from_reward(rewards, self.gamma)
        sg = self._stat_gamma
        self.count += 1
        sg = sg * (1 - 1 / self.count)
        mean = returns[to_include].mean().detach()
        std = returns[to_include].std().detach()
        if not torch.isnan(mean + std):
            self.mean = self.mean * sg + (1 - sg) * mean.item()
            self.std = self.std * sg + (1 - sg) * (std.item())
        returns = (returns - self.mean) / (self.std + 1e-8)
        # compute advantages
        advantages = returns - value_estimates
        # Calculate the critic loss, only where actions are random
        if is_random is not None:
            masked_td = advantages * is_random.float()
        else:
            masked_td = advantages
        critic_loss = torch.square(masked_td)
        # critic_loss.register_hook(lambda grad: print("Grad C Loss", torch.abs(grad).sum()))
        # Calculate the actor loss incorporating the entropy term
        actor_loss = -1 * log_probs * advantages.detach() - self.alpha * entropies
        if to_include is not None:
            actor_loss = actor_loss[to_include]
            critic_loss = critic_loss[to_include]
        actor_loss = actor_loss.sum()
        critic_loss = critic_loss.sum()
        if torch.isnan(critic_loss + actor_loss):
            print("NAN, returns", returns)
            print("NAN, V", value_estimates)
            print("NAN, Prob", log_probs)
        return critic_loss, actor_loss