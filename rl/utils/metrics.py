#!/usr/bin/env python

from collections import deque

class ValueHistory(object):
    
    def __init__(self, history_size):
        self.history = deque(maxlen=history_size)
        
    def push(self, data):
        self.history.append(data)
        
    def mean(self):
        if len(self.history) == 0:
            return None
        
        return sum(self.history) / len(self.history)
    
    def max(self):
        if len(self.history) == 0:
            return None
        
        return max(self.history)