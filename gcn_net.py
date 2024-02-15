import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCNNet(torch.nn.Module):
    def __init__(self, num_nodes, num_classes, dropout_rate=0.3):
        super(GCNNet, self).__init__()
        self.conv1 = GCNConv(num_nodes, 64)
        self.conv2 = GCNConv(64, 64)
        self.conv3 = GCNConv(64, num_classes)
        self.dropout_rate = dropout_rate

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        # x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        # x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.conv3(x, edge_index)
        # x = F.dropout(x, p=self.dropout_rate, training=self.training)
        return F.softmax(x, dim=1)
