# network.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GATv2Model(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=2, return_attention=False):
        super(GATv2Model, self).__init__()
        self.return_attention = return_attention
        from torch_geometric.nn import GATv2Conv
        self.gat_conv1 = GATv2Conv(in_channels, hidden_channels, heads=heads, dropout=0.1, edge_dim=1)
        self.gat_conv2 = GATv2Conv(hidden_channels * heads, out_channels, heads=1, dropout=0.1, edge_dim=1)
    
    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        if self.return_attention:
            x, attn_weights = self.gat_conv1(x, edge_index, edge_attr, return_attention_weights=True)
        else:
            x = self.gat_conv1(x, edge_index, edge_attr)
            attn_weights = None
        x = F.elu(x)
        x = self.gat_conv2(x, edge_index, edge_attr)
        if self.return_attention:
            return x, attn_weights
        else:
            return x

def plot_attention_matrix(attn_weights, title="Attention Matrix", plot=True):
    """
    Plot the attention matrix as a heatmap.
    attn_weights: a tuple (edge_index, alpha) returned by the first GATv2 layer's forward call.
    """
    if not plot or attn_weights is None:
        return
    import matplotlib.pyplot as plt
    edge_index, alpha = attn_weights
    # Average over heads if multiple exist.
    attn_avg = alpha.mean(dim=1).cpu().numpy()
    num_nodes = int(edge_index.max().item() + 1)
    attn_matrix = np.zeros((num_nodes, num_nodes))
    edge_index_np = edge_index.cpu().numpy()
    for idx, (i, j) in enumerate(edge_index_np.T):
        attn_matrix[i, j] = attn_avg[idx]
    plt.figure(figsize=(8, 6))
    plt.imshow(attn_matrix, cmap="viridis")
    plt.colorbar()
    plt.title(title)
    plt.xlabel("Node")
    plt.ylabel("Node")
    plt.show()
