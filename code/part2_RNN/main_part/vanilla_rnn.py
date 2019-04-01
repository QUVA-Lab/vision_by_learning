################################################################################
# MIT License
#
# Copyright (c) 2018
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2018
# Date Created: 2018-09-04
################################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

################################################################################

class VanillaRNN(nn.Module):

    def __init__(self, seq_length, input_dim, num_hidden, num_classes, batch_size, device='cuda:0'):

        super(VanillaRNN, self).__init__()

        self._seq_length = seq_length
        self._input_dim = input_dim
        self._num_hidden = num_hidden
        self._num_classes = num_classes
        self._batch_size = batch_size
        self._device = torch.device(device) if isinstance(device, str) else device

        # Initialize the weights and biases
        self._Wx, _ = self.init_layer(self._input_dim, self._num_hidden, use_bias=False)
        self._Wh, self._bh = self.init_layer(self._num_hidden, self._num_hidden)
        self._Wo, self._bo = self.init_layer(self._num_hidden, self._num_classes)

        # Weight initialization
        self.reset_parameters()


    def forward(self, x):

        x = x.permute(1,0,2)  # [seq_length, batch_size, 1]
        h = torch.zeros([self._batch_size, self._num_hidden])
        h = h.to(self._device)

        # Step through time for RNN
        for x_step in x:
            Wxx = torch.matmul(x_step, self._Wx)
            Wxh = torch.matmul(h, self._Wh) + self._bh
            h = torch.tanh(Wxx + Wxh)

        # Output layer
        h = (h @ self._Wo) + self._bo
        return F.log_softmax(h, dim=1)

    def init_layer(self, input_dim, output_dim, use_bias=True):
        # Basically implements nn.Linear
        W = Parameter(torch.empty(input_dim, output_dim, device=self._device, requires_grad=True))
        b = None
        if use_bias:
            b = Parameter(torch.empty(output_dim, device=self._device, requires_grad=True))
        return W, b

    def reset_parameters(self):
        stdv = 1.0/math.sqrt(self._num_hidden)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
