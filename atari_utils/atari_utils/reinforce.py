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
                 lr=1e-4,
                 clip_param=0.1,
                 value_loss_coef=0.5,
                 num_mini_batch=4,
                 entropy_coef=0.01,
                 KL_loss_coef=0.01,
                 eps=1e-5,
                 max_grad_norm=0.5,
                 epoch=4
                 ):

        self.env = env
        self.device = device
        self.gamma = gamma
        self.lr = lr
        self.clip_param = clip_param
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
        self.KL_loss_coef = KL_loss_coef
        self.max_grad_norm = max_grad_norm
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr, eps=eps)

    def learn(
            self,
            steps,
            eval_env_name=None,
            eval_policy_wrapper=PolicyWrapper,
            eval_episodes=30,
            eval_agents=None,
            verbose=True,
            evaluations=10,
            graph=False,
            use_ppo_lr_decay=False,
            use_clipped_value_loss=True,
            score_training=True,
            logger=None
    ):
        if eval_agents is None:
            eval_agents = self.env.num_envs

        num_updates = steps // self.num_steps // self.num_processes

        self.use_clipped_value_loss = use_clipped_value_loss
        
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
        if verbose:
            iterator = trange(num_updates, desc='Training agent', unit_scale=self.num_steps * self.num_processes)
        else:
            iterator = range(num_updates)

        for j in iterator:
            for step in range(self.num_steps):
                # Sample actions
                with torch.no_grad():
                    value, action, action_log_prob = self.actor_critic.act(rollouts.obs[step], full_log_prob=True)

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
            
            if (j % ceil(num_updates / evaluations) == 0 or j == num_updates - 1) and eval_env_name is not None:
                eval_metrics = evaluate(  # TODO add steps tracking
                    eval_policy_wrapper(self),
                    eval_env_name,
                    self.device,
                    agents=eval_agents,
                    episodes=eval_episodes,
                    verbose=False
                )
                eval_mean_scores.append(eval_metrics['eval_score_mean'])
                eval_mean_scores_std.append(eval_metrics['eval_score_std'])

            metrics = {
            'ppo_value_loss': float(value_loss),
            'ppo_action_loss': float(action_loss)
            }
            if score_queue and score_training:
                metrics.update({'score_mean': moving_average_score[-1]})
            if eval_mean_scores:
                metrics.update({'eval_score_mean': eval_mean_scores[-1]})
            if logger is not None:
                logger.log(metrics)

        return eval_mean_scores


    def update(self, rollouts):
        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0

        for e in range(self.epoch):
            data_generator = rollouts.feed_forward_generator(None, self.num_mini_batch)

            for sample in data_generator:
                obs_batch, actions_batch, value_preds_batch, return_batch, \
                masks_batch, old_full_log_probs_batch, adv_targ = sample

                # Reshape to do in a single forward pass for all steps
                values, action_log_probs, dist_entropy = self.actor_critic.evaluate_actions(obs_batch, actions_batch)
                _, full_log_probs, _ = self.actor_critic.evaluate_actions(obs_batch, actions_batch, True)

                action_loss = -torch.mean(action_log_probs * return_batch)
                
                pi_theta_batch = torch.distributions.Categorical(full_log_probs)
                pi_theta_old_batch = torch.distributions.Categorical(old_full_log_probs_batch)
                KL_loss = torch.distributions.kl.kl_divergence(pi_theta_batch, pi_theta_old_batch).mean()

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
                
                '''
                this loss is equivalent to SAC
                '''
                self.optimizer.zero_grad()
                (value_loss * self.value_loss_coef + action_loss + KL_loss * self.KL_loss_coef).backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                         self.max_grad_norm)
                self.optimizer.step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()
            
        num_updates = self.epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch
    

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
