#!/usr/bin/env python
# -*- encoding: utf-8 -*-

"""
Implementation of the Deep Q-Network.

Deep Q-Networks can (should) only be used on discrete action spaces, because
it is non-trivial to find the argmax (the best action) over a continuous set
of possible actions.

Source: 
- Mnih, Volodymyr, et al. “Playing Atari with Deep Reinforcement Learning.”
  ArXiv:1312.5602 [Cs], Dec. 2013. arXiv.org, http://arxiv.org/abs/1312.5602.
"""

from gym import spaces
from functools import reduce
from .utils.replay import ReplayBuffer
from .utils.metrics import MovingAverage

import random
import copy

import torch
from torch.nn.utils import clip_grad_norm_

class DQN(object):
    
    def __init__(self, env, model_fn, opt_cls, opt_config):
        self.env = env
        
        if not isinstance(self.env.action_space, spaces.Discrete):
            raise Exception(
                "Only environments with `Discrete` action "
                "spaces can be solved with DQN."
                )
        
        self.num_actions = self.env.action_space.n
        self.obs_dim = reduce(lambda a, b: a * b, self.env.observation_space.shape, 1)
        print(f"[env]: num_actions={self.num_actions}")
        print(f"[env]: obs_dim={self.obs_dim}")
        
        self.model = model_fn(self.obs_dim, self.num_actions)
        
        self.target_model = model_fn(self.obs_dim, self.num_actions)
        self.target_model.load_state_dict(self.model.state_dict())
        self._freeze_weights(self.target_model)
        
        self.optimizer = opt_cls(self.model.parameters(), **opt_config)
        
        self.replay_buffer = None
        self.return_moving_average = None
    
    def _freeze_weights(self, model):
        for param in model.parameters():
            param.requires_grad = False
    
    def _reshape(self, replay):
        obs, actions, rewards, done, next_obs = list(zip(*replay))
        
        obs = torch.tensor(obs, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        done = torch.tensor(done, dtype=torch.float32)
        next_obs = torch.tensor(next_obs, dtype=torch.float32)
        
        return obs, actions, rewards, done, next_obs
        
    def train(self,
              steps_train_interval=1,
              steps_warmup=200,
              batch_size=128,
              discount_factor=0.999,
              epsilon_fn=lambda ep: 0.90 * (2 ** (-ep / 50)) + 0.05,
              replay_buffer_size=10000,
              episodes_target_update=5,
              num_episodes=5000,
              track_average=100,
              pretrain_hook=None,
              posttrain_hook=None):
        """
        Args:
          - num_episodes (int):
          - track_average (int):
          - pretrain_hook (Function):
          - posttrain_hook (Function):
        """
        
        self.replay_buffer = ReplayBuffer(replay_buffer_size)
        self.return_moving_avg = MovingAverage(track_average)
        
        # total number of steps taken
        steps = 0
        
        for ep in range(1, num_episodes + 1):
            ep_return = 0
            ep_steps = 0
            
            obs = self.env.reset()
            done = False
            
            while not done:
                with torch.no_grad():
                    action = self.env.action_space.sample()
                    
                    if random.random() > epsilon_fn(ep):
                        # placed in [obs] for a batch size of 1
                        action_logits = self.model(
                            torch.tensor(
                                [obs],
                                dtype=torch.float32
                            )) 

                        # get the best action for the first entry in the batch
                        # (there is only 1 element in the batch)
                        action = (torch
                            .argmax(action_logits, dim=1)[0]
                            .cpu()
                            .numpy()
                            .item())
                
                next_obs, reward, done, info = self.env.step(action)
                ep_return += reward
                
                self.replay_buffer.add_batch([(
                    obs,
                    action,
                    reward,
                    1. if done else 0.,
                    next_obs
                )])
                
                steps += 1
                ep_steps += 1
                obs = next_obs
            
                # train
                if (steps + 1) % steps_train_interval == 0:
                    if steps < steps_warmup:
                        print(f"[opt][step {steps}], still warming up.")
                    else:
                        self.optimizer.zero_grad()
                        
                        if pretrain_hook is not None:
                            pretrain_hook(self, locals())

                        replay = self.replay_buffer.sample(batch_size)
                        b_obs, b_acts, b_rwds, b_done, b_next_obs = self._reshape(replay)
                        
                        target_logits = self.target_model(b_next_obs)

                        max_a = torch.max(target_logits, dim=1).values
                        td_target = b_rwds + discount_factor * max_a * (1 - b_done)

                        q_est = self.model(b_obs)
                        q_est_indexed = torch.gather(q_est, 1, b_acts.unsqueeze(1))

                        loss = torch.mean((td_target - q_est_indexed) ** 2)
                        loss.backward()
                        
                        clip_grad_norm_(self.model.parameters(), 1.0)
                        
                        self.optimizer.step()

                        if posttrain_hook is not None:
                            posttrain_hook(self, locals())
            
            self.return_moving_avg.push(ep_return)
                        
            if (ep + 1) % episodes_target_update == 0:
                # update the target network
                self.target_model.load_state_dict(self.model.state_dict())
                self._freeze_weights(self.target_model)
                
                print(f"[ep {ep}] return={ep_return}, "
                      f"avg_return={self.return_moving_avg.value()}")
                        
                    