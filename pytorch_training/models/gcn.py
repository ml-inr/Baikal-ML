import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCN(nn.Module):
    def __init__(self, in_features, hidden_size, out_size, dropout_p=0.0):
        super().__init__()
        self.conv1 = GCNConv(in_features, hidden_size)
        self.dropout1 = nn.Dropout(p=dropout_p)
        self.conv2 = GCNConv(hidden_size, hidden_size)
        self.dropout2 = nn.Dropout(p=dropout_p)
        self.conv3 = GCNConv(hidden_size, hidden_size)
        self.dropout3 = nn.Dropout(p=dropout_p)
        self.conv4 = GCNConv(hidden_size, out_size)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.dropout1(x)
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        x = self.dropout2(x)
        x = self.conv3(x, edge_index)
        x = F.elu(x)
        x = self.dropout3(x)
        x = self.conv4(x, edge_index)
        return x
