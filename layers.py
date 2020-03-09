import math

import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


"""
Define a flatten layer to convert input feature set from 3D to 2D (from [800,235,196] to [800,235*196])
no need.
"""

class Flatten(Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


    
"""
Define a avg pooling layer to average the 
"""
class ave_pooling(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self):
        super(ave_pooling, self).__init__()

    
    def forward(self,input):
        output=torch.zeros([input.shape[0],input.shape[-1]], device = "cuda:0")
        for i in range(input.shape[0]):
            output[i]=input[i,::].mean(0)
        
        return output
    
    def __repr__(self):
        return self.__class__.__name__    
    

class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        output = torch.zeros(input.shape[0], input.shape[1], self.out_features, device = "cuda:0")
        for i in range(input.shape[0]):
            support = torch.mm(input[i].double(), self.weight.double())
            output[i] = torch.spmm(adj[i], support.double())
            if self.bias is not None:
                for j in range(input.shape[1]):
                    output[i, j] = output[i, j] + self.bias             
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
