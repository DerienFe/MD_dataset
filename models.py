import torch.nn as nn
import torch.nn.functional as F
from layers import *


class GCN(nn.Module):
    def __init__(self, nfeat, nhid1, nhid2, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid1)
        self.gc2 = GraphConvolution(nhid1, nhid2)
        self.dropout = dropout
        self.ave_pooling=ave_pooling()
        self.linear = nn.Linear(nhid2,1)
        

    def forward(self, x, adj):
        #x = nn.Flatten()
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj))
        x = self.ave_pooling(x)
        x = self.linear(x)
        return x

