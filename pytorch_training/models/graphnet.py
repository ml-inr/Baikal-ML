import torch
import torch.nn as nn
from torch_geometric.nn import knn_graph
import torch_geometric.nn as gnn


class DynEdgeConv(gnn.EdgeConv):
    def __init__(
        self,
        nn: nn.Module,
        aggr: str = "max",
        nb_neighbors: int = 4,
        features_subset=None,
        **kwargs,
    ):
        if features_subset is None:
            features_subset = slice(None)  # Use all features
        assert isinstance(features_subset, (list, slice))

        super().__init__(nn=nn, aggr=aggr, **kwargs)

        self.nb_neighbors = nb_neighbors
        self.features_subset = features_subset

    def forward(self, x, edge_index, batch=None):
        x = super().forward(x, edge_index)

        edge_index = knn_graph(
            x=x[:, self.features_subset],
            k=self.nb_neighbors,
            batch=batch,
        ).to(x.device)

        return x, edge_index


class GraphnetDynedge(nn.Module):
    def __init__(
        self,
        in_features,
        knn_neighbours=4,
        dynedge_layer_sizes=None,
        out_size=2,
        second_head_out_size=None,
        aggregate_output=False,
    ):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._dynedge_layer_sizes = dynedge_layer_sizes or [
            (128, 512),
            (768, 512),
            (768, 512),
            (768, 512),
        ]
        self._conv_layers = nn.ModuleList()
        self._activation = nn.ReLU()
        self._nb_neigbours = knn_neighbours
        self._post_processing_layer_sizes = [512, 256]
        self._global_pooling_schemes = ["min", "max", "mean"]
        self.readout_layers_sizes = [128]
        self.aggregate_output = aggregate_output

        nb_input_features = in_features
        nb_latent_features = nb_input_features
        for sizes in self._dynedge_layer_sizes:
            layers = []
            layer_sizes = [nb_latent_features] + list(sizes)

            for ix, (nb_in, nb_out) in enumerate(
                zip(layer_sizes[:-1], layer_sizes[1:])
            ):
                if ix == 0:
                    nb_in *= 2
                layers.append(nn.Linear(nb_in, nb_out))
                layers.append(self._activation)

            сonv_layer = DynEdgeConv(
                nn.Sequential(*layers), aggr="add", nb_neighbors=self._nb_neigbours
            )
            self._conv_layers.append(сonv_layer)
            nb_latent_features = nb_out

        nb_latent_features = (
            sum(sizes[-1] for sizes in self._dynedge_layer_sizes) + nb_input_features
        )
        # print()
        post_processing_layers = []
        layer_sizes = [nb_latent_features] + list(self._post_processing_layer_sizes)

        for nb_in, nb_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            post_processing_layers.append(nn.Linear(nb_in, nb_out))
            post_processing_layers.append(self._activation)

        self._post_processing = nn.Sequential(*post_processing_layers)

        # nb_poolings = len(self._global_pooling_schemes)
        # nb_latent_features = nb_out * nb_pooling
        self.head = nn.Linear(256, out_size)

    def forward(self, x, edge_index, batch):
        skip_connections = [x]
        for conv_layer in self._conv_layers:
            x, edge_index = conv_layer(x, edge_index, batch)
            skip_connections.append(x)

        x = torch.cat(skip_connections, dim=1)
        y = self._post_processing(x)
        z = self.head(y)
        if self.aggregate_output:
            return gnn.global_mean_pool(z, batch)

        return z


class Encoder(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_size,
        num_layers,
        dim_feedforward_size,
        n_heads,
        out_size,
        dropout_p,
        aggregator=None,
        second_head_out_size=None,
        **kwargs,
    ):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.first_layer = nn.Linear(in_features, hidden_size)
        enc_layer = nn.TransformerEncoderLayer(
            hidden_size, n_heads, dim_feedforward_size, dropout_p, batch_first=True
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers)
        self.head = nn.Linear(hidden_size, out_size)
        self.second_head = (
            nn.Linear(hidden_size, second_head_out_size)
            if second_head_out_size is not None
            else None
        )
        self.aggregator = aggregator

    def forward(self, x, mask):
        x = self.first_layer(x)
        x = self.enc(x, src_key_padding_mask=~mask)
        y = self.head(x)
        if self.aggregator is not None:
            return self.aggregator(x)
        if self.second_head is not None:
            z = self.second_head(x)
            return torch.cat([y, z], dim=-1)
        return y


class GraphnetAndEncoderStack(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder = Encoder(**kwargs["encoder_params"])
        self.graphnet = GraphnetDynedge(**kwargs["graphnet_params"])

    def forward(self, x, mask):
        x = self.graphnet(x, mask)
        return self.encoder(x, mask)
