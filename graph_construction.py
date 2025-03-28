# graph_construction.py
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors


def compute_edge_weights(coords, node_features, k=5, alpha=0.5, beta=0.5):
    """
    Given an array of global coordinates (N,2) and node_features (N, D),
    compute a k-nearest neighbor graph.
    For each edge, compute two similarities:
      - Spatial similarity: 1 / (1 + distance)
      - Visual similarity: cosine similarity between node feature vectors.
    Combine them with: weight = alpha * spatial_sim + beta * visual_sim.
    Returns:
      - edge_index: tensor of shape [2, num_edges]
      - edge_weight: tensor of shape [num_edges]
    """
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(coords)
    distances, indices = nbrs.kneighbors(coords)
    N = coords.shape[0]
    edge_index_list = []
    edge_weight_list = []

    # Normalize node features along dimension 1 for cosine similarity.
    node_features_norm = F.normalize(node_features, p=2, dim=1)

    for i in range(N):
        for idx, j in enumerate(indices[i][1:]):  # skip self
            spatial_sim = 1.0 / (1.0 + distances[i][idx + 1])
            vis_sim = torch.dot(node_features_norm[i], node_features_norm[j]).item()
            weight = alpha * spatial_sim + beta * vis_sim
            edge_index_list.append([i, j])
            edge_weight_list.append(weight)

    edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(edge_weight_list, dtype=torch.float)
    return edge_index, edge_weight


def create_graph_data(node_features, edge_index, edge_attr, ground_truth):
    """
    Create a PyTorch Geometric Data object given node features, edge_index,
    edge attributes (edge weights), and ground truth labels.
    """
    from torch_geometric.data import Data
    data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, y=ground_truth)
    return data
