from collections import deque
from math import ceil

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import trange

from a2c_ppo_acktr.policy import Policy
from a2c_ppo_acktr.rollout_storage import RolloutStorage
from atari_utils.evaluation import evaluate
from atari_utils.policy_wrappers import PolicyWrapper

class REINFORCE:

    def __init__(self,
                 env,
                 device,
                 num_steps=100,
                 gamma=0.99,
                 lr=3e-4,
                 clip_param=0.1,
                 value_loss_coef=0.5,
                 num_mini_batch=4,
                 entropy_coef=0.01,
                 eps=1e-5,
                 max_grad_norm=0.5,
                 epoch=4
                 ):

        self.env = env
        self.device = device
        self.gamma = gamma
        self.lr = lr
        self.num_steps = num_steps
        if hasattr(self.env, 'num_envs'):
            self.num_processes = self.env.num_envs
        else:
            self.num_processes = 1

        self.actor_critic = None
        self.reset_actor_critic()

        self.epoch = epoch
        self.num_mini_batch = num_mini_batch
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr, eps=eps)

    def learn(
            self,
            eval_env_name=None,
            eval_policy_wrapper=PolicyWrapper,
            eval_episodes=30,
            eval_agents=None,
            evaluations=10,
            graph=False,
            score_training=True,
            logger=None
    ):
        if eval_agents is None:
            eval_agents = self.env.num_envs
        
        rollouts = RolloutStorage(
            self.num_steps,
            self.num_processes,
            self.env.observation_space.shape,
            self.env.action_space
        )

        obs = self.env.reset()
        rollouts.obs[0].copy_(obs)
        rollouts.to(self.device)
        score_queue = deque(maxlen=10)
        moving_average_score = []
        eval_mean_scores = []
        eval_mean_scores_std = []

        for step in range(self.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob = self.actor_critic.act(rollouts.obs[step])

            # Obser reward and next obs
            obs, reward, done, infos = self.env.step(action)

            for info in infos:
                if 'r' in info.keys():
                    score_queue.append(info['r'])
                    moving_average_score.append(np.mean(score_queue))

            # If done then clean the history of observations.
            masks = (~torch.tensor(done)).float().unsqueeze(1)
            rollouts.insert(obs, action, action_log_prob, value, reward, masks)

        with torch.no_grad():
            next_value = self.actor_critic.get_value(rollouts.obs[-1]).detach()

        rollouts.compute_returns(next_value, False, self.gamma, None)
        value_loss, action_loss, dist_entropy = self.update(rollouts)


    def update(self, rollouts):
        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0

        for e in range(self.epoch):
            data_generator = rollouts.feed_forward_generator(None, self.num_mini_batch)

            for sample in data_generator:
                obs_batch, actions_batch, value_preds_batch, return_batch, \
                masks_batch, old_action_log_probs_batch, adv_targ = sample

                # Reshape to do in a single forward pass for all steps
                values, action_log_probs, dist_entropy = self.actor_critic.evaluate_actions(obs_batch, actions_batch)

                action_loss = - torch.mean(action_log_probs * return_batch)
                
                '''
                #Possible combination: use clipped ratio and future return

                ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
                surr1 = ratio * return_batch
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * return_batch
                action_loss = -torch.min(surr1, surr2).mean()
                '''

                '''
                still need to implement  -log_pi(a_t | s_t) * sum_{t'=t}^T log(pi(a_t' | s_t') / pi_old(a_t' | s_t'))
                '''


                if self.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch + \
                                         (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (
                            value_pred_clipped - return_batch).pow(2)
                    value_loss = 0.5 * torch.max(value_losses,
                                                 value_losses_clipped).mean()
                else:
                    value_loss = 0.5 * (return_batch - values).pow(2).mean()

                self.optimizer.zero_grad()
                (value_loss * self.value_loss_coef + action_loss -
                 dist_entropy * self.entropy_coef).backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                         self.max_grad_norm)
                self.optimizer.step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()
    

    def set_env(self, env):
        self.env = env
        self.num_processes = env.num_envs

    def save(self, path):
        torch.save(self.actor_critic, path)

    def load(self, path):
        self.actor_critic = torch.load(path)

    def act(self, obs, full_log_prob=False):
        return self.actor_critic.act(obs, deterministic=True, full_log_prob=full_log_prob)

    def reset_actor_critic(self):
        self.actor_critic = Policy(self.env.observation_space.shape, self.env.action_space)
        self.actor_critic.to(self.device)
