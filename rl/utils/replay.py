#!/usr/bin/env python

import copy

from collections import deque

class ReplayBuffer(object):
    
    def __init__(self, buffer_size):
        self.buffer = deque(maxlen=buffer_size)
        
    def add_batch(self, batch):
        for b in batch:
            self.buffer.append(copy.deepcopy(b))
    
    def sample(self, size):
        items = random.sample(self.buffer, size)