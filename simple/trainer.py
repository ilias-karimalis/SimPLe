import os

import torch
import torch.nn as nn
from torch.cuda import empty_cache
from torch.nn.utils import clip_grad_norm_
from tqdm import trange
from itertools import product

from atari_utils.logger import WandBLogger
from simple.adafactor import Adafactor
from matplotlib import animation
import matplotlib.pyplot as plt



class Trainer:

    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.logger = None
        if self.config.use_wandb:
            self.logger = WandBLogger()

        self.optimizer = Adafactor(self.model.parameters())

        self.model_step = 1
        self.reward_step = 1
        
        self.last_model_frames_pred = None
        self.last_frames_input = None
    
    
    """
    Here, we want KL(P_true,P_model) = sum_i(P_true(x_i)*log(P_true(x_i)/P_model(x_i))
    where i = 1...256 for a single pixel. This is computed for 3*105*80 pixel locations
    where we consider P_true to be a degenerate categorical distribution with probability
    1 on its true pixel value and zero elsewhere, that is a 256 dimension 1 hot vector.
    We consider P_model as a categorical parameterized by the 256 logits output by the
    world model. Then, we can do the simiplification
    
    KL(P_true,P_model) = P_true(x_i)*log(P_true(x_i)/P_model(x_i))I(x_i = x_true)
                       = P_true(x_true)*log(P_true(x_true)/P_model(x_true))
                       = 1*log(1/P_model(x_true))
                       = -1*log(P_model(x_true))
    
    Our network outputs logit(p_i) = log(p_i/(1-p_i)) per pixel, so regretably we will
    need to compute the probability p_i then log it.
    
    TODO: I think maximizing logit(p) is equivalent to maximizing log(p), but possible not at
    the same rate, will need to think about it as it would be very computationally convenient.
    
    Aside:
    Yes this is equivalent to cross entroy given the degenerate categorical has zero entropy
    but in other cases this would not be true.
    """
    def totally_not_cross_entropy_KL_loss(self, frames_pred, frames_true):
        frames_pred = torch.softmax(torch.permute(frames_pred, (0,2,3,4,1)), dim = 4)
        return -1*torch.log(torch.gather(frames_pred, dim = 4, index = frames_true.unsqueeze(-1)).squeeze(-1))
        
        
##       More readable but significantly slower
#        divergence = torch.zeros_like(frames_true, dtype=torch.float64)
#        for b in range(self.config.batch_size):
#            for c,i,j in product(range(3), range(105),range(80)):
#                divergence[b,c,i,j] = -1*torch.log(frames_pred[b,c,i,j,frames_true[b,c,i,j]])
#        return divergence


    """
    TODO: what is going on here hmmmm
    Here, we want to ensure that our model does not take drastic steps away from where it currently is.
    To do this we ensure that the KL divergence between distributions of generated from the same input
    frames is not too large by encorporating it into the loss for the model parameter update. We weight
    this KL by a beta factor to ensure that the main goal of the model is to indeed model the true enviroment,
    but also will not have high variance from step to step. This reminds me a lot of TRPO but for the model
    instead of the policy and thus I have thrown trust region around. The math here involves just computing
    the KL between two categorical distributions that have been generated by the ith model and the i-1th model.
    The following code rearranges the Tensors a bit, converts logits to probabilities, then uses entrywise operations
    and a summation for the pixel-wise KL divergence. We can then post process this KL matrix however we like.
    """
    def trust_region_loss(self, frames_pred_new, frames_pred_old):
        frames_pred_new = torch.clip(torch.softmax(torch.permute(frames_pred_new, (0,2,3,4,1)), dim = 4), 1e-10, 1)
        frames_pred_old = torch.clip(torch.softmax(torch.permute(frames_pred_old, (0,2,3,4,1)), dim = 4), 1e-10, 1)
        return torch.sum(torch.mul(frames_pred_new, torch.log(frames_pred_new) - torch.log(frames_pred_old)), dim = 4)
        
    
    def train(self, epoch, env, result_directory, steps=15000, render_rollout=False):
        if epoch == 0:
            steps *= 3
            
        render_frames = []
        true_frames = []

        c, h, w = self.config.frame_shape
        rollout_len = self.config.rollout_length
        states, actions, rewards, new_states, dones, values = env.buffer[0]

        if env.buffer[0][5] is None:
            raise BufferError('Can\'t train the world model, the buffer does not contain one full episode.')

        assert states.dtype == torch.uint8
        assert actions.dtype == torch.uint8
        assert rewards.dtype == torch.uint8
        assert new_states.dtype == torch.uint8
        assert values.dtype == torch.float32

        def get_index():
            index = -1
            while index == -1:
                index = int(torch.randint(len(env.buffer) - rollout_len, size=(1,)))
                for i in range(rollout_len):
                    done, value = env.buffer[index + i][4:6]
                    if done or value is None:
                        index = -1
                        break
            return index

        def get_indices():
            return [get_index() for _ in range(self.config.batch_size)]

        def preprocess_state(state):
            state = state.float() / 255
            noise_prob = torch.tensor([[self.config.input_noise, 1 - self.config.input_noise]])
            noise_prob = torch.softmax(torch.log(noise_prob), dim=-1)
            noise_mask = torch.multinomial(noise_prob, state.numel(), replacement=True).view(state.shape)
            noise_mask = noise_mask.to(state)
            state = state * noise_mask + torch.median(state) * (1 - noise_mask)
            return state

        reward_criterion = nn.CrossEntropyLoss()

        iterator = trange(
            0,
            steps,
            rollout_len,
            desc='Training world model',
            unit_scale=rollout_len
        )
        for i in iterator:
            # Scheduled sampling
            if epoch == 0:
                decay_steps = self.config.scheduled_sampling_decay_steps
                inv_base = torch.exp(torch.log(torch.tensor(0.01)) / (decay_steps // 4))
                epsilon = inv_base ** max(decay_steps // 4 - i, 0)
                progress = min(i / decay_steps, 1)
                progress = progress * (1 - 0.01) + 0.01
                epsilon *= progress
                epsilon = 1 - epsilon
            else:
                epsilon = 0

            indices = get_indices()
            frames = torch.empty((self.config.batch_size, c * self.config.stacking, h, w))
            frames = frames.to(self.config.device)

            for j in range(self.config.batch_size):
                frames[j] = env.buffer[indices[j]][0].clone()
                if render_rollout: # only works with batch size 1
                    render_frames.append(torch.permute(env.buffer[indices[j]][0].clone()[0:3], (1,2,0)).cpu())
                    render_frames.append(torch.permute(env.buffer[indices[j]][0].clone()[3:6], (1,2,0)).cpu())
                    render_frames.append(torch.permute(env.buffer[indices[j]][0].clone()[6:9], (1,2,0)).cpu())
                    render_frames.append(torch.permute(env.buffer[indices[j]][0].clone()[9:], (1,2,0)).cpu())
                    true_frames.append(torch.permute(env.buffer[indices[j]][0].clone()[0:3], (1,2,0)).cpu())
                    true_frames.append(torch.permute(env.buffer[indices[j]][0].clone()[3:6], (1,2,0)).cpu())
                    true_frames.append(torch.permute(env.buffer[indices[j]][0].clone()[6:9], (1,2,0)).cpu())
                    true_frames.append(torch.permute(env.buffer[indices[j]][0].clone()[9:], (1,2,0)).cpu())

            frames = preprocess_state(frames)

            n_losses = 6 if self.config.use_stochastic_model else 5
            losses = torch.empty((rollout_len, n_losses))

            if self.config.stack_internal_states:
                self.model.init_internal_states(self.config.batch_size)

            for j in range(rollout_len):
                _, actions, rewards, new_states, _, values = env.buffer[0]
                actions = torch.empty((self.config.batch_size, *actions.shape))
                actions = actions.to(self.config.device)
                rewards = torch.empty((self.config.batch_size, *rewards.shape), dtype=torch.long)
                rewards = rewards.to(self.config.device)
                new_states = torch.empty((self.config.batch_size, *new_states.shape), dtype=torch.long)
                new_states = new_states.to(self.config.device)
                values = torch.empty((self.config.batch_size, *values.shape))
                values = values.to(self.config.device)
                for k in range(self.config.batch_size):
                    actions[k] = env.buffer[indices[k] + j][1]
                    rewards[k] = env.buffer[indices[k] + j][2]
                    new_states[k] = env.buffer[indices[k] + j][3]
                    values[k] = env.buffer[indices[k] + j][5]

                new_states_input = new_states.float() / 255

                self.model.train()
                frames_pred, reward_pred, values_pred = self.model(frames, actions, new_states_input, epsilon)
                # Note, this reward_pred is a classifier output, hence being a 3 vector classsifying on {-1,0,1}
                

                if j < rollout_len - 1:
                    for k in range(self.config.batch_size):
                        if float(torch.rand((1,))) < epsilon:
                            frame = new_states[k]
                        else:
                            frame = torch.argmax(frames_pred[k], dim=0)
                        frame = preprocess_state(frame)
                        frames[k] = torch.cat((frames[k, c:], frame), dim=0)

#                loss_reconstruct = nn.CrossEntropyLoss(reduction='none')(frames_pred, new_states)
                loss_reconstruct = self.totally_not_cross_entropy_KL_loss(frames_pred, new_states)
                
                if render_rollout:
                    render_frames.append(torch.permute(torch.argmax(frames_pred[0], dim=0), (1,2,0)).cpu())
                    true_frames.append(torch.permute(new_states[0], (1,2,0)).cpu())

                
                clip = torch.tensor(self.config.target_loss_clipping).to(self.config.device)
                loss_reconstruct = torch.max(loss_reconstruct, clip)
                loss_reconstruct = loss_reconstruct.mean() - self.config.target_loss_clipping
                
                loss_trust_region = 0.0
                if j > 1:
                    frames_pred_new, _, _ = self.model(*self.last_frames_input)
                    frames_pred_old = self.last_model_frames_pred
                    loss_trust_region = self.config.trust_region_beta*self.trust_region_loss(frames_pred_new, frames_pred_old).mean()
#                    print(loss_trust_region)
                

                loss_value = nn.MSELoss()(values_pred, values)
                loss_reward = reward_criterion(reward_pred, rewards)
                loss = loss_reconstruct + loss_value + loss_reward + loss_trust_region
                
                loss_lstm = 0.0
                if self.config.use_stochastic_model:
                    loss_lstm = self.model.stochastic_model.get_lstm_loss()
                    loss = loss + loss_lstm
                
                if not torch.is_tensor(frames_pred):
                    frames_pred = torch.tensor(frames_pred)
                if not torch.is_tensor(frames):
                    frames = torch.tensor(frames)
                if not torch.is_tensor(actions):
                    actions = torch.tensor(actions)
                if not torch.is_tensor(new_states_input):
                    new_states_input = torch.tensor(new_states_input)
                if not torch.is_tensor(epsilon):
                    epsilon = torch.tensor(epsilon)

                self.last_model_frames_pred = frames_pred.detach()
                self.last_frames_input = (frames.detach(), actions.detach(), new_states_input.detach(), epsilon.detach())


#                torch.autograd.set_detect_anomaly(True)

                self.optimizer.zero_grad()
                loss.backward()
                


                clip_grad_norm_(self.model.parameters(), self.config.clip_grad_norm, error_if_nonfinite=True)

                self.optimizer.step()

                tab = [float(loss), float(loss_reconstruct), float(loss_value), float(loss_reward), float(loss_trust_region)]
                if self.config.use_stochastic_model:
                    tab.append(float(loss_lstm))
                losses[j] = torch.tensor(tab)
                
                if render_rollout:
                    path='gifs/'
                    filename=f'hallucinations_{i}.gif'
                    #Mess with this to change frame size
                    plt.figure(figsize=(render_frames[0].shape[1] / 36.0, render_frames[0].shape[0] / 36.0), dpi=200)

                    patch = plt.imshow(render_frames[0])
                    plt.axis('off')

                    def animate(i):
                        patch.set_data(render_frames[i])

                    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(render_frames), interval=50)
                    anim.save(path + filename, writer='imagemagick', fps=30)
                    
                    path='./'
                    filename=f'truth_{i}.gif'
                    #Mess with this to change frame size
                    plt.figure(figsize=(true_frames[0].shape[1] / 36.0, true_frames[0].shape[0] / 36.0), dpi=200)

                    patch = plt.imshow(true_frames[0])
                    plt.axis('off')

                    def animate(i):
                        patch.set_data(true_frames[i])

                    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(true_frames), interval=50)
                    anim.save(path + filename, writer='imagemagick', fps=30)


            losses = torch.mean(losses, dim=0)
            metrics = {
                'loss': float(losses[0]),
                'loss_reconstruct': float(losses[1]),
                'loss_value': float(losses[2]),
                'loss_reward': float(losses[3]),
                'loss_trust_region': float(losses[5])
            }
            if self.config.use_stochastic_model:
                metrics.update({'loss_lstm': float(losses[4])})

            if self.logger is not None:
                d = {'model_step': self.model_step, 'epsilon': epsilon}
                d.update(metrics)
                self.logger.log(d)
                self.model_step += rollout_len

            iterator.set_postfix(metrics)
            



        empty_cache()
        if self.config.save_models:
            torch.save(self.model.state_dict(), os.path.join(result_directory, 'model.pt'))
