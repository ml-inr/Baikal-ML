import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from torch_geometric.nn import GCNConv


def get_mlp(dim_in, dim_out):
    return nn.Sequential(
        gnn.Linear(dim_in, dim_out), nn.ReLU(), gnn.Linear(dim_out, dim_out)
    )


class GINCN(nn.Module):
    def __init__(self, in_features, hidden_size, out_size, dropout_p=0.0):
        super().__init__()
        self.conv1 = gnn.GINEConv(nn=get_mlp(in_features, hidden_size))
        self.dropout1 = nn.Dropout(p=dropout_p)
        self.conv2 = gnn.GINEConv(nn=get_mlp(hidden_size, hidden_size))
        self.dropout2 = nn.Dropout(p=dropout_p)
        self.conv3 = gnn.GINEConv(nn=get_mlp(hidden_size, out_size))

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.dropout1(x)
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        x = self.dropout2(x)
        x = self.conv3(x, edge_index)
        return x
