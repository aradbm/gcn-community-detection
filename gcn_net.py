import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

'''
Here we have GCNNet class that inherits from torch.nn.Module.

The GCN architecture is used, with the following layers:
- GCNConv layer with input size of num_nodes and output size of 64
- GCNConv layer with input size of 64 and output size of 32
- GCNConv layer with input size of 32 and output size of num_classes

'''


class GCNNet(torch.nn.Module):
    def __init__(self, num_nodes, num_classes):
        super(GCNNet, self).__init__()
        self.conv1 = GCNConv(num_nodes, 64)
        self.conv2 = GCNConv(64, 32)
        self.conv3 = GCNConv(32, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)

        return F.softmax(x, dim=1)
