# network.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Data

# Import our modules.
from deep_features import load_json_data, merge_nodes, build_node_features
from graph_construction import compute_edge_weights, create_graph_data

# ---------------------
# Configuration
# ---------------------
import os
# Base data path (adjust as needed)
data_base = "/home/fuad/Work/Projects/hurricane/data"

# We assume the validation split is stored under:
val_json_dir = os.path.join(data_base, "val", "JSON")
val_frame_dir = os.path.join(data_base, "val", "FRAME")
val_mask_dir = os.path.join(data_base, "val", "MASK")


# Transformer model for visual features
model_name = "MITLL/LADI-v2-classifier-large-reference"
pooling_mode = "mean"  # "mean" for advanced pooling or "cls" for using CLS token

# Hyperparameters for edge weights.
alpha = 0.5  # spatial similarity weight
beta = 0.5   # visual similarity weight
k_neighbors = 5  # number of neighbors for KNN graph

# ---------------------
# Define GATv2 Model
# ---------------------
class GATv2Model(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=2):
        super(GATv2Model, self).__init__()
        self.conv1 = GATv2Conv(in_channels, hidden_channels, heads=heads, dropout=0.1, edge_dim=1)
        self.conv2 = GATv2Conv(hidden_channels * heads, out_channels, heads=1, dropout=0.1, edge_dim=1)
    
    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.conv1(x, edge_index, edge_attr)
        x = F.elu(x)
        x = self.conv2(x, edge_index, edge_attr)
        return x

# ---------------------
# Main Testing Script
# ---------------------
def main():
    # Load transformer model and processor.
    from transformers import AutoImageProcessor, AutoModelForImageClassification
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForImageClassification.from_pretrained(model_name, output_hidden_states=True)
    model.eval()
    
    # Load and merge raw nodes from JSON annotations.
    raw_nodes = load_json_data(val_json_dir)
    print(f"Loaded {len(raw_nodes)} raw annotations from validation set.")
    nodes = merge_nodes(raw_nodes)
    print(f"Merged into {len(nodes)} unique building nodes.")
    
    # Extract global coordinates for graph construction.
    import numpy as np
    coords = np.array([node['global_coordinate'] for node in nodes])
    
    # Build node features (concatenating deep visual features with metadata).
    node_feats, gt_damage = build_node_features(nodes, val_frame_dir, val_mask_dir, processor, model, pooling_mode=pooling_mode)
    print(f"Extracted node feature matrix of shape: {node_feats.shape}")
    
    # Compute edge weights based on spatial and visual similarities.
    edge_index, edge_weight = compute_edge_weights(coords, node_feats, k=k_neighbors, alpha=alpha, beta=beta)
    print(f"Constructed graph with {edge_index.shape[1]} edges.")
    
    # Create PyTorch Geometric Data object.
    data = create_graph_data(node_feats, edge_index, edge_weight, gt_damage)
    print("Graph data object created.")
    
    # Initialize GATv2 model.
    in_dim = node_feats.shape[1]
    num_classes = 6  # damage states: 0 to 5
    model_gat = GATv2Model(in_channels=in_dim, hidden_channels=128, out_channels=num_classes, heads=2)
    model_gat.eval()
    
    with torch.no_grad():
        out = model_gat(data)
    
    preds = out.argmax(dim=1)
    
    print("Predicted damage states for each node:")
    for i, pred in enumerate(preds):
        print(f"Building {nodes[i]['building_id']}: Predicted: {pred.item()} | Ground Truth: {nodes[i]['damage_state']}")
    
if __name__ == "__main__":
    main()
