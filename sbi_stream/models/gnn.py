
from typing import List, Dict, Any, Tuple, Callable

import torch
import torch.nn as nn
import torch_geometric.nn as gnn
import torch_geometric.transforms as T

from sbi_stream.models import utils
from sbi_stream.models import mlp

class GNNBlock(nn.Module):
    def __init__(self,
        input_size: int, output_size: int, layer_name: str,
        layer_params: Dict[str, Any] = None, activation_fn: Callable = nn.ReLU(),
        layer_norm: bool = False, norm_first: bool = False
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.layer_name = layer_name
        self.layer_params = layer_params or {}
        self.activation_fn = activation_fn
        self.layer_norm = layer_norm
        self.norm_first = norm_first
        self.has_edge_attr = False
        self.has_edge_weight = False
        self.graph_layer = None
        self.norm = None

        self._setup_model()

    def _setup_model(self):
        if self.layer_name == "ChebConv":
            self.has_edge_attr = False
            self.has_edge_weight = True
            self.graph_layer =  gnn.ChebConv(
                self.input_size, self.output_size, **self.layer_params)
        elif self.layer_name == "GCNConv":
            self.has_edge_attr = False
            self.has_edge_weight = True
            self.graph_layer =  gnn.GCNConv(
                self.input_size, self.output_size, **self.layer_params)
        elif self.layer_name == "SAGEConv":
            self.has_edge_attr = False
            self.has_edge_weight = False
            self.graph_layer =  gnn.SAGEConv(
                self.input_size, self.output_size, **self.layer_params)
        elif self.layer_name == "GATConv":
            self.has_edge_attr = True
            self.has_edge_weight = False
            self.layer_params['concat'] = False  # only works when False
            self.graph_layer =  gnn.GATConv(
                self.input_size, self.output_size, **self.layer_params)
        else:
            raise ValueError(f"Unknown graph layer: {layer_name}")

        if self.layer_norm:
            self.norm = gnn.norm.LayerNorm(self.output_size)

    def forward(self, x, edge_index, edge_attr=None, edge_weight=None):
        if self.has_edge_attr:
            x = self.graph_layer(x, edge_index, edge_attr)
        elif self.has_edge_weight:
            x = self.graph_layer(x, edge_index, edge_weight)
        else:
            x = self.graph_layer(x, edge_index)
        if self.norm_first and self.norm is not None:
            x = self.norm(x)
            x = self.activation_fn(x)
        elif self.norm is not None:
            x = self.activation_fn(x)
            x = self.norm(x)
        else:
            x = self.activation_fn(x)
        return x


class GNN(nn.Module):
    """ Embedding model based on Graph Neural Networks (GNNs). """

    def __init__(
        self, input_size: int, hidden_sizes: List[int], projection_size: int = None,
        graph_layer: str = "ChebConv", graph_layer_params: Dict[str, Any] = None,
        activation_name: str = "relu", activation_args: Dict[str, Any] = None,
        pooling: str = "mean", layer_norm: bool = False, norm_first: bool = False
    ):
        """ Initialize the GNN embedding model.
        Parameters
        ----------
        input_size : int
            The size of the input features.
        hidden_sizes : list of int
            The sizes of the hidden layers.
        projection_size : int, optional
            The size of the projection layer. Default: None
        graph_layer : str, optional
            The type of graph layer to use. Default: "ChebConv"
        graph_layer_params : dict, optional
            Additional parameters for the graph layer. Default: None
        activation_name : str, optional
            The name of the activation function to use. Default: "relu"
        activation_args : dict, optional
            Additional arguments for the activation function. Default: None
        pooling : str, optional
            The type of global pooling to use. Default: "mean"
        layer_norm : bool, optional
            Whether to use layer normalization. Default: False
        norm_first : bool, optional
            Whether to apply normalization before activation. Default: False
        """

        super().__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.projection_size = projection_size
        self.graph_layer = graph_layer
        self.graph_layer_params = graph_layer_params or {}
        self.activation_fn = utils.get_activation(activation_name, activation_args)
        self.pooling = pooling
        self.layer_norm = layer_norm
        self.norm_first = norm_first
        self.layers = None
        self.has_edge_attr = None
        self.has_edge_weight = None

        # setup the model
        if self.projection_size:
            self.projection_layer = nn.Linear(self.input_size, self.projection_size)
            input_size = self.projection_size
        else:
            self.projection_layer = None
            input_size = self.input_size

        self.layers = nn.ModuleList()
        layer_sizes = [input_size] + self.hidden_sizes
        for i in range(1, len(layer_sizes)):
            layer = GNNBlock(
                layer_sizes[i-1], layer_sizes[i], self.graph_layer,
                self.graph_layer_params, self.activation_fn,
                self.layer_norm, self.norm_first
            )
            self.layers.append(layer)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor,
        edge_attr: torch.Tensor = None, edge_weight: torch.Tensor = None
    ) -> torch.Tensor:
        if self.projection_layer:
            x = self.projection_layer(x)

        for layer in self.layers:
            x = layer(x, edge_index, edge_attr, edge_weight)

        # global pooling
        if self.pooling == "mean":
            return gnn.global_mean_pool(x, batch)
        elif self.pooling == "max":
            return gnn.global_max_pool(x, batch)
        elif self.pooling == "sum":
            return gnn.global_add_pool(x, batch)
        else:
            return x


class GNNEmbedding(nn.Module):
    """ GNN + MLP embedding model. """

    def __init__(self, gnn_args: Dict[str, Any], mlp_args: Dict[str, Any]):
        """ Initialize the GNN + MLP embedding model.
        Parameters
        ----------
        gnn_args : dict
            Arguments for the GNN model.
        mlp_args : dict
            Arguments for the MLP model.
        """

        super().__init__()
        self.gnn_args = gnn_args
        self.mlp_args = mlp_args
        self.gnn = GNN(
            input_size=gnn_args['input_size'],
            hidden_sizes=gnn_args['hidden_sizes'],
            projection_size=gnn_args.get('projection_size', None),
            graph_layer=gnn_args.get('graph_layer', 'ChebConv'),
            graph_layer_params=gnn_args.get('graph_layer_params', {}),
            activation_name=gnn_args.get('activation_name', 'relu'),
            activation_args=gnn_args.get('activation_args', {}),
            pooling=gnn_args.get('pooling', 'mean'),
            layer_norm=gnn_args.get('layer_norm', False),
            norm_first=gnn_args.get('norm_first', False)
        )
        self.mlp = mlp.MLPBatchNorm(
            input_size=self.gnn.hidden_sizes[-1],
            output_size=mlp_args['output_size'],
            hidden_sizes=mlp_args.get('hidden_sizes', []),
            activation_fn=utils.get_activation(
                mlp_args.get('activation_name', 'relu'),
                mlp_args.get('activation_args', {})
            ),
            batch_norm=mlp_args.get('batch_norm', False),
            dropout=mlp_args.get('dropout', 0.0)
        )

    def forward(self, batch_dict):
        """ Forward pass through the GNN embedding model. GNN -> MLP
        Parameters
        ----------
        batch_dict : dict
            A dictionary containing the batch data with keys:
            - 'x': Node features tensor of shape (num_nodes, num_node_features)
            - 'edge_index': Edge indices tensor of shape (2, num_edges)
            - 'batch': Batch vector tensor of shape (num_nodes,)
            - 'edge_attr' (optional): Edge attributes tensor of shape (num_edges, num_edge_features)
            - 'edge_weight' (optional): Edge weights tensor of shape (num_edges,)
        Returns
        -------
        torch.Tensor
            The output embeddings of shape (batch_size, output_size)
        """
        x = batch_dict['x']
        edge_index = batch_dict['edge_index']
        batch = batch_dict['batch']
        edge_attr = batch_dict.get('edge_attr', None)
        edge_weight = batch_dict.get('edge_weight', None)

        x = self.gnn(x, edge_index, batch, edge_attr=edge_attr, edge_weight=edge_weight)
        x = self.mlp(x)
        return x