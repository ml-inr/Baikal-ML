import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv


class GAT(nn.Module):
    def __init__(self, in_features, hidden_size, out_size=2, dropout_p=0.0, heads=1):
        super().__init__()
        # for debug purposes
        self.in_features = in_features
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.heads = heads
        self.gat1 = GATv2Conv(in_features, hidden_size, heads=heads)
        self.dropout1 = nn.Dropout(p=dropout_p)
        self.gat2 = GATv2Conv(hidden_size * heads, out_size, heads=1)

    def forward(self, x, edge_index, batch):
        # edge_inds = torch.arange(x.shape[0]).to(self.device)
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = self.dropout1(x)
        x = self.gat2(x, edge_index)
        return x
