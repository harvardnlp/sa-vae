#!/usr/bin/env python3

import sys
import os

import argparse
import json
import random
import shutil

import torch
from torch import cuda
import torch.nn as nn

import torch.nn.functional as F
import numpy as np

class Optim:
  def __init__(self, lr=0.1, method='adam', state=None):
    self.lr = lr
    self.eps = 1e-6
    self.method = method
    self.state = {}
    if method == 'adam':
      self.optim_params = {'beta1': 0.9, 'beta2': 0.999}
    elif method == 'mom':
      self.optim_params = {'beta': 0}

  def reset(self):
    self.state = {}
    
  def step(self, x, dfdx):
    if self.method == 'adam':
      beta1 = self.optim_params['beta1']
      beta2 = self.optim_params['beta2']
      if 't' not in self.state:
        self.state['t'] = 0
        self.state['m'] = torch.zeros(x.size()).type_as(x)
        self.state['v'] = torch.zeros(x.size()).type_as(x)
        self.state['denom'] = torch.zeros(x.size()).type_as(x)
      self.state['t'] += 1
      self.state['m'].mul_(beta1).add_(1-beta1, dfdx)
      self.state['v'].mul_(beta2).addcmul_(1-beta2, dfdx, dfdx)
      self.state['denom'].copy_(self.state['v']).sqrt_().add_(self.eps)
      bias1 = 1 - beta1**self.state['t']
      bias2 = 1 - beta2**self.state['t']
      return x - self.lr*self.state['m']/self.state['denom']
    
    elif self.method == 'mom':
      beta = self.optim_params['beta']
      if 'm' not in self.state:
        self.state['m'] = torch.zeros(x.size()).type_as(x)
      self.state['m'].mul_(beta).add_(-1, dfdx)
      return x + self.lr*self.state['m']
    
    elif self.method == 'sgd':
      return x - self.lr*dfdx
      
    
