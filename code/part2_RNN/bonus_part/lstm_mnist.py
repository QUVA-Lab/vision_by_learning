import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Recurrent neural network (many-to-one)
class LSTM_MNIST(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
      """
      Initializes LSTM_MNIST object. 

      Args:
        input_size: size of input vector.
        hidden_size: size of the hidden state.
        num_layers: how many layers for hidden or cell states.
        num_classes: number of output classes.

      TODO:
      Implement initialization of the network.
      """
      
      ########################
      # PUT YOUR CODE HERE  #
      #######################
      raise NotImplementedError
      ########################
      # END OF YOUR CODE    #
      #######################
    
    def forward(self, x):
      """
      Performs forward pass of the input. Here an input tensor x.
      You are expected to do 3 steps:
       - set initial hidden and cell states 
       - forward propagate LSTM
       - decode the hidden state of the last time step

      Args:
        x: input to the network
      Returns:
        out: outputs of the network

      TODO:
      Implement forward pass of the network.
      """

      ########################
      # PUT YOUR CODE HERE  #
      #######################
      raise NotImplementedError
      ########################
      # END OF YOUR CODE    #
      #######################
        
      return out
