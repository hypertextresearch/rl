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
from .metrics import MovingAverage

import random
import copy

class DQN(object):
    
    def __init__(self, env, model_fn, opt_cls, opt_config):
        self.env = env
        
        self.model = model_fn()
        
        self.target_model = model_fn()
        self.target_model.load_state_dict(self.model.state_dict())
        # Hold target model weights fixed.
        self._freeze_weights(self.target_model)
        
        self.optimizer = opt_cls(self.model.parameters(), **opt_config)
        
        if instanceof(self.env.action_space, spaces.Discrete):
            raise Exception(
                "Only environments with `Discrete` action "
                "spaces can be solved with DQN."
                )
        
        self.num_actions = self.env.n
        
        self.replay_buffer = None
        self.return_moving_average = None
    
    def _freeze_weights(self, model):
        for param in model.parameters():
            param.requires_grad = False
    
    def _reshape(self, replay):
        obs, actions, rewards, done, next_obs = list(zip(replay))
        return obs, actions, rewards, next_obs
        
    def train(self,
              steps_train_interval=50,
              batch_size=1024,
              discount_factor=0.9,
              epsilon=0.05,
              replay_buffer_size=1000000,
              num_episodes=1000,
              track_average=30
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
            
            obs, reward, done, info = self.env.reset()
            
            while not done:
                with torch.no_grad():
                    action = self.env.action_space.sample()
                    
                    if random.random() > epsilon:
                        # placed in [obs] for a batch size of 1
                        action_logits = self.model(torch.tensor([obs])) 

                        # get the best action for the first entry in the batch
                        # (there is only 1 element in the batch)
                        action = torch.argmax(action_logits, dim=1)[0]
                
                next_obs, reward, done, info = env.step(action)
                
                self.replay_buffer.push([(
                    obs,
                    action,
                    reward,
                    done,
                    next_obs
                )])
                
                steps += 1
                obs = next_obs
            
                # train
                if steps > steps_warmup and (steps + 1) % steps_train_interval == 0:
                    self.optimizer.zero_grad()

                    replay = self.replay_buffer.sample(batch_size)
                    b_obs, b_acts, b_rwds, b_done, b_next_obs = self._reshape(replay)

                    max_a = torch.max(self.target_model(b_obs), dim=1)
                    td_target = b_rwds + discount_factor * max_a * b_done

                    # TODO: check if this even works (gather)
                    q_est = self.model(b_obs)
                    q_est_indexed = torch.gather(q_est, 1, b_acts)

                    loss = torch.mean((td_target - q_est_indexed) ** 2)

                    loss.backward()
                    self.optimizer.step()

                    self.target_model.load_state_dict(self.model.state_dict())
                    self._freeze_weights(self.target_model)
                    