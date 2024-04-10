"""
   some simple cnn 
"""
import torch
import torch.nn as nn


def my_conv(in_channels, out_channels, kernel_size):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, padding='same'),
        nn.LeakyReLU(inplace=True)
        )
           
class CNN(nn.Module):
    def __init__(self, in_channels=2, out_channels=2, depth=5, hidden_dim=64, kernel_size=5, dropout=0.):
        super(CNN, self).__init__()
        self.depth = depth
        self.dropout = dropout
        self.conv_in = my_conv(in_channels, hidden_dim, kernel_size)
        self.conv_hidden = nn.ModuleList([my_conv(hidden_dim, hidden_dim, kernel_size) for _ in range(self.depth-2)]) 
        self.conv_out = my_conv(hidden_dim, out_channels, kernel_size)

    def forward(self, x):
        x = self.conv_in(x)
        for layer in self.conv_hidden:
            x = layer(x)
        x = self.conv_out(x)
        return x
       
def simple_cnn(params, **kwargs):
    model = CNN(in_channels=params.in_chan, out_channels=params.out_chan, depth=params.depth, hidden_dim=64,
                kernel_size=3, **kwargs)
    return model
