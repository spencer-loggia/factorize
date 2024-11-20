import pickle

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from reward_functions import ActorCritic
from RNNAgent import Agent
from scipy.ndimage import gaussian_filter
from neurotools import util
import pickle


class _HCartPole:
    """
    class to wrap non-vectorized env so it matches.
    """
    def __init__(self):
        self.env = gym.make("CartPole-v1", render_mode="human")

    def reset(self):
        observations, infos = self.env.reset()
        return observations[None, :], infos

    def step(self, action):
        observations, rewards, terminations, truncations, infos = self.env.step(action.squeeze())
        return observations[None, ...], np.array(rewards), np.array(terminations), np.array(truncations), infos


class Environment:
    """
    class to hold agen, track statistics, and connect to environment
    """
    def __init__(self, agent, batch_size=50, use_velocities=False, lr=.01, device="cpu"):
        self.device = device
        if use_velocities:
            self.input_size = 4
        else:
            self.input_size = 2
        self.env_name = "CartPole-v1"
        self.n_env = batch_size
        self.env = gym.make_vec(self.env_name, num_envs=self.n_env, vectorization_mode="async")
        self.human_env = _HCartPole()
        self.agent = agent # the RNN agent that generates actions.
        self.critic = torch.nn.Sequential(
            torch.nn.Linear(4, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1)
        ) # simply value prediction for training. i.e. linear
        # model that project future val at each time point/
        self.lr = lr

    def play(self, human=False):
        self.agent.reset()
        if human:
            env = self.human_env
            n_env = 1
        else:
            env = self.env
            n_env = self.n_env
        observations, infos = env.reset()
        over = np.array([False] * n_env)
        true_rewards = []
        all_obs = []
        all_actions = []
        log_probs = []
        critic_exp_reward = []
        entropies = []
        overs = []
        while False in over:
            stim = torch.from_numpy(observations).to(self.device) # <n_env, input_size>
            if self.input_size == 2:
                stim = stim[:, 0::2]
            out = self.agent.step(stim).view((len(stim), -1)) # send current obsevation to agent, get out <n_env, 1>
            action_prob_dist = torch.sigmoid(out) # probability of going right.
            action_prob_dist = torch.concatenate([1 - action_prob_dist, action_prob_dist], dim=1)
            dist = torch.distributions.Categorical(probs=action_prob_dist) # left(0) or right (1)
            entropy = dist.entropy()
            actions = dist.sample()
            # get current reward and states and next observations from environments
            # what was the apriori probability of the action that occured.
            log_ap = dist.log_prob(actions)
            # get the expected reward from the critic
            # critic_in = torch.concatenate([torch.from_numpy(observations).to(self.device),
            #                                actions.view((-1, 1)).detach()], dim=1)
            critic_in = torch.from_numpy(observations).to(self.device)
            exp_r = self.critic(critic_in)
            all_obs.append(observations)
            observations, rewards, terminations, truncations, infos = env.step(actions.detach().cpu().numpy())
            local_done = np.logical_or(truncations, terminations)
            over = np.logical_or(over, local_done)
            overs.append(over)
            true_rewards.append(torch.tensor(rewards).to(self.device))
            log_probs.append(log_ap)
            all_actions.append(out.detach().cpu().numpy())
            critic_exp_reward.append(exp_r)
            entropies.append(entropy)
        log_probs = torch.stack(log_probs, dim=0)
        true_rewards = torch.stack(true_rewards, dim=0)
        critic_exp_reward = torch.stack(critic_exp_reward, dim=0).view((len(true_rewards), -1))
        entropies = torch.stack(entropies, dim=0)
        overs = np.stack(overs)
        all_obs = np.stack(all_obs)
        all_actions = np.stack(all_actions)
        to_include = torch.from_numpy(np.logical_not(overs)).to(self.device)
        lifespans = np.argmax(overs.astype(int), axis=0)
        return true_rewards, log_probs, critic_exp_reward, entropies, to_include, lifespans, all_obs, all_actions

    def train(self, iterations=1000):
        critic_loss_hist = []
        actor_loss_hist = []
        lifespans = []
        act_optimizer = torch.optim.Adam(lr=self.lr, params=list(self.critic.parameters()))
        crit_optimizer = torch.optim.Adam(lr=self.lr, params=list(self.agent.parameters()))
        a3c_loss = ActorCritic(gamma=.98, alpha=0.1)

        for i in range(iterations):
            print("train epoch", i)
            act_optimizer.zero_grad()
            crit_optimizer.zero_grad()
            self.agent.train = True
            gt_r, log_probs, exp_r, H, to_include, lifespan, _, _ = self.play()
            lifespans.append(lifespan)
            critic_loss, actor_loss = a3c_loss(gt_r, exp_r, log_probs, H, to_include)
            actor_loss = actor_loss
            # a3c_loss.gamma = .3 * min(1 - .999**(i), .999) + .7
            #a3c_loss.alpha = a3c_loss.alpha * .9993
            if i < 250:
                a = 0.
            else:
                a = 1.
            # loss = 1.0 * a * actor_loss + 1. * critic_loss
            critic_loss.backward()
            actor_loss.backward()
            critic_loss_hist.append(critic_loss.detach().cpu().item())
            actor_loss_hist.append(actor_loss.detach().cpu().item())
            print("C loss:", critic_loss_hist[-1], "A loss:", actor_loss_hist[-1])
            crit_optimizer.step()
            act_optimizer.step()
            if i % 1000 == 0:
                self.agent.train = False
                plt.plot(np.stack(lifespans).mean(axis=1))
                plt.show()
                self.play(human=True)
            #optimizer, done = util.is_converged(loss_hist, optimizer, self.n_env, i)
            # if done:
            #     break
        return critic_loss_hist, actor_loss_hist, np.stack(lifespans).mean(axis=1)


if __name__=="__main__":
    iter = 7500
    agent = Agent(input_size=2, latent_size=16)
    with open("models/gru_factor_8.pkl", "rb") as f:
        agent = pickle.load(f)
    env = Environment(agent, batch_size=1, lr=.001)
    chist, ahist, lifespan = env.train(iter)
    chist = gaussian_filter(np.array(chist), sigma=5.)
    ahist = gaussian_filter(np.array(ahist), sigma=5.)
    lifespan = gaussian_filter(lifespan, sigma=3.)
    fig, ax = plt.subplots(3)
    ax[0].plot(chist)
    ax[1].plot(ahist)
    ax[2].plot(lifespan)
    plt.show()
    env.play(human=True)
    with open("models/gru_factor_8.pkl", "wb") as f:
        pickle.dump(agent, f)
    print("done!")



# env = gym.make("CartPole-v1", render_mode="human")
# observation, info = env.reset()
#
# episode_over = False
# while not episode_over:
#     action = env.action_space.sample()  # agent policy that uses the observation and info
#     observation, reward, terminated, truncated, info = env.step(action)
#
#     episode_over = terminated or truncated
#
# env.close()