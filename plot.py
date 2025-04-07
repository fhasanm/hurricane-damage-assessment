# main.py
import os
import torch
import numpy as np
import random

# Import our modules.
from dataloader import load_split, load_and_build
from graph_construction import compute_edge_weights, create_graph_data, visualize_graph
from deep_features import plot_deep_features_collage_for_buildings
from network import GATv2Model, plot_attention_matrix

from transformers import AutoImageProcessor, AutoModelForImageClassification

# ---------------------
# Configuration
# ---------------------
data_base = "/home/fuad/Work/Projects/hurricane/data"
split = "val"  

model_name = "MITLL/LADI-v2-classifier-large-reference"
pooling_mode = "mean"  # "mean" or "cls"

# Hyperparameters for edge weight computation.
alpha = 0.5      # spatial similarity weight
beta = 0.5       # visual similarity weight
k_neighbors = 5  # number of neighbors for KNN graph

# Plotting toggles.
PLOT_DEEP_FEATURES = False  
PLOT_GRAPH = False          
PLOT_ATTENTION = True      

# Transformer checkpoint toggle.
USE_CHECKPOINTED_TRANSFORMER = False  # Set to False to use the frozen pretrained model
CHECKPOINT_DIR = "checkpoints"                
CHECKPOINT_FILE_PREFIX = "best_transformer_checkpoint_epoch_"

# For inference we typically use the frozen transformer.
fine_tune_transformer = False

def get_latest_transformer_checkpoint(ckpt_dir, prefix):
    """
    Scan the specified directory for transformer checkpoints matching the given prefix,
    extract the epoch number, and return the path to the checkpoint with the highest epoch.
    Returns None if no matching files are found.
    """
    if not os.path.isdir(ckpt_dir):
        return None

    max_epoch = -1
    latest_ckpt_path = None
    for fname in os.listdir(ckpt_dir):
        if fname.startswith(prefix) and fname.endswith(".pt"):
            try:
                part = fname.replace(prefix, "").replace(".pt", "")
                epoch_num = int(part)
                if epoch_num > max_epoch:
                    max_epoch = epoch_num
                    latest_ckpt_path = os.path.join(ckpt_dir, fname)
            except ValueError:
                pass
    return latest_ckpt_path

def main():
    # Load the validation split using the global dataloader.
    nodes, coords, frame_dir, mask_dir = load_split(data_base, split)
    print(f"Loaded and merged {len(nodes)} unique building nodes from {split} split.")

    # Initialize the local transformer model for feature extraction.
    processor_local = AutoImageProcessor.from_pretrained(model_name)
    model_local = AutoModelForImageClassification.from_pretrained(
        model_name, 
        output_hidden_states=True
    )

    if USE_CHECKPOINTED_TRANSFORMER:
        latest_ckpt = get_latest_transformer_checkpoint(CHECKPOINT_DIR, CHECKPOINT_FILE_PREFIX)
        if latest_ckpt is not None:
            print(f"Loading transformer checkpoint from: {latest_ckpt}")
            ckpt_data = torch.load(latest_ckpt, map_location="cpu")
            model_local.load_state_dict(ckpt_data["transformer_state_dict"], strict=False)
        else:
            print("No transformer checkpoint found. Using the frozen pretrained model instead.")
    else:
        print("Using the frozen pretrained transformer model.")

    # Optionally, plot a collage of building patches and their PCA-based embeddings.
    if PLOT_DEEP_FEATURES and len(nodes) > 0:
        plot_deep_features_collage_for_buildings(
            nodes,
            frame_dir=frame_dir,
            mask_dir=mask_dir,
            processor=processor_local,
            model=model_local,
            pooling_mode=pooling_mode,
            n_examples=4,
            plot=True
        )

    # Load full validation data using the global dataloader.
    val_data = load_and_build(data_base, split, processor_local, model_local, pooling_mode, fine_tune_transformer)
    # Ensure consistency by updating nodes and coords.
    nodes = val_data["nodes"]
    coords = val_data["coords"]
    node_feats = val_data["node_feats"]
    gt_damage = val_data["gt_damage"]

    print(f"Extracted node feature matrix of shape: {node_feats.shape}")

    # Compute edge weights (combining spatial and visual similarity).
    edge_index, edge_weight = compute_edge_weights(
        coords, 
        node_feats,
        k=k_neighbors,
        alpha=alpha,
        beta=beta
    )
    print(f"Constructed graph with {edge_index.shape[1]} edges.")

    # Create PyTorch Geometric Data object.
    data = create_graph_data(node_feats, edge_index, edge_weight, gt_damage)
    print("Graph data object created.")
    
    # *** FIX: Move the data object to the correct device ***
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = data.to(device)

    # (Optional) Visualize graph.
    if PLOT_GRAPH:
        final_nodes = []
        for i, node in enumerate(nodes):
            final_nodes.append({
                'node_id': i,
                'building_id': node['building_id'],
                'global_coordinate': node['global_coordinate'],
                'damage_state': node['damage_state']
            })
        visualize_graph(final_nodes, coords, None, {}, plot=True, filename="dorianet_graph.html")

    # Example model usage: Create and run a GATv2 model.
    in_dim = node_feats.shape[1]
    num_classes = 6  # damage states 0-5

    # Update the hyperparameters to match the training checkpoint.
    model_gat = GATv2Model(
        in_channels=in_dim, 
        hidden_channels=256,   
        out_channels=num_classes,
        heads=4,               
        return_attention=True
    )
    CHECKPOINT_PATH = "./checkpoints/best_model_seed122_20250404_115442.pt"
    model_gat.to(device)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model_gat.load_state_dict(checkpoint["model_gat_state_dict"])
    model_gat.eval()
    with torch.no_grad():
        out, attn_weights = model_gat(data)
    preds = out.argmax(dim=1)

    print("Predicted damage states for each node (example):")
    for i, pred in enumerate(preds):
        print(f"Building {nodes[i]['building_id']}: Predicted: {pred.item()} | Ground Truth: {nodes[i]['damage_state']}")

    # Plot attention matrix.
    plot_attention_matrix(attn_weights, title="GATv2 Layer 1 Attention", plot=PLOT_ATTENTION)

if __name__ == "__main__":
    main()
