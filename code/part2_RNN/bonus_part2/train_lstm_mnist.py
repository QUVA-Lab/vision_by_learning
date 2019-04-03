"""
This module implements training and evaluation of a LSTM in PyTorch for classifying MNIST.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from lstm_mnist import LSTM_MNIST

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
sequence_length = 28
input_size = 28
hidden_size = 128
num_layers = 2
num_classes = 10
batch_size = 100
num_epochs = 2
learning_rate = 0.01

# MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='../../data/',
                                           train=True, 
                                           transform=transforms.ToTensor(),
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='../../data/',
                                          train=False, 
                                          transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size, 
                                          shuffle=False)

model = LSTM_MNIST(input_size, hidden_size, num_layers, num_classes).to(device)


# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
      
        # Here, you're expected to do 3 steps:
        # 1. Change the image from 2D matrix into a 1D vector (sequence of pixels)
        # 2. Forward pass of the input
        # 3. Backward pass and optimize
        # 4. Print the loss and accuracy at the end of the training epoch.
      
        ########################
        # PUT YOUR CODE HERE  #
        #######################
        raise NotImplementedError
        ########################
        # END OF YOUR CODE    #
        #######################
      
# Test the model
with torch.no_grad():
    correct = 0
    total = 0
    
    ########################
    # PUT YOUR CODE HERE  #
    #######################
    raise NotImplementedError
    ########################
    # END OF YOUR CODE    #
    #######################
