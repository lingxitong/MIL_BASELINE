import torch
import torch.nn as nn
from utils.model_utils import get_act

class RNN_MIL(nn.Module):

    def __init__(self, in_dim,n_dim,num_classes,act = 'relu'):
        super(RNN_MIL, self).__init__()
        self.n_dim = n_dim

        self.fc1 = nn.Linear(in_dim, n_dim)
        self.fc2 = nn.Linear(n_dim, n_dim)

        self.fc3 = nn.Linear(n_dim, num_classes)

        self.activation = get_act(act)

    def forward(self, input, state):
        input = self.fc1(input)
        state = self.fc2(state)
        state = self.activation(state+input)
        output = self.fc3(state)
        return output, state

    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.ndims)