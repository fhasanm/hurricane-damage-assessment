#dataloader.py
import os
import numpy as np
from deep_features import load_json_data, merge_nodes, build_node_features

def load_split(data_base, split):
    """
    Load and merge JSON files for a given split ('train' or 'val').
    Returns:
      - nodes: merged building nodes.
      - coords: numpy array of global coordinates.
      - frame_dir: directory containing frame images.
      - mask_dir: directory containing mask images.
    """
    json_dir = os.path.join(data_base, split, "JSON")
    frame_dir = os.path.join(data_base, split, "FRAME")
    mask_dir = os.path.join(data_base, split, "MASK")
    
    raw_nodes = load_json_data(json_dir)
    nodes = merge_nodes(raw_nodes)
    coords = np.array([n['global_coordinate'] for n in nodes])
    return nodes, coords, frame_dir, mask_dir

def load_and_build(data_base, split, processor, model, pooling_mode, fine_tune_transformer):
    """
    For a given split, load the JSON data, merge it, and then build node features.
    Returns a dictionary with:
       - nodes: merged nodes.
       - coords: global coordinates.
       - frame_dir, mask_dir: directories for images.
       - node_feats: node features.
       - gt_damage: ground truth damage labels.
    """
    nodes, coords, frame_dir, mask_dir = load_split(data_base, split)
    node_feats, gt_damage = build_node_features(
        nodes, frame_dir, mask_dir,
        processor, model,
        pooling_mode=pooling_mode,
        fine_tune_transformer=fine_tune_transformer
    )
    return {
         "nodes": nodes,
         "coords": coords,
         "frame_dir": frame_dir,
         "mask_dir": mask_dir,
         "node_feats": node_feats,
         "gt_damage": gt_damage
    }
