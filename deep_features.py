# deep_features.py
import os
import json
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModelForImageClassification
from collections import Counter

def load_json_data(json_dir):
    """
    Load all JSON files from the given directory and return a list of raw building annotations.
    Each JSON file corresponds to one video frame.
    """
    json_files = sorted([f for f in os.listdir(json_dir) if f.endswith('.json')])
    raw_nodes = []
    for jf in json_files:
        json_path = os.path.join(json_dir, jf)
        with open(json_path, 'r') as f:
            data = json.load(f)
        frame_name = data.get('Frame_Name', None)
        buildings = data.get('Buildings', [])
        for b in buildings:
            try:
                coord = np.array(np.matrix(b[1])).flatten()  # e.g., [latitude, longitude]
                if coord.size < 2:
                    continue
            except Exception as e:
                print(f"Error parsing coordinate for building {b[0]} in file {jf}: {e}")
                continue
            raw_nodes.append({
                'building_id': b[0],
                'global_coordinate': [float(coord[0]), float(coord[1])],
                'mask_filename': b[2],
                'damage_state': int(b[3]),
                'num_stories': int(b[4]),
                'annotation_effort': int(b[5]) if len(b) > 5 else None,
                'frame': frame_name
            })
    return raw_nodes

def merge_nodes(raw_nodes):
    """
    Merge nodes with the same building_id (i.e. the same building annotated in multiple frames).
    Merging rules:
      - Global coordinates: averaged.
      - Damage state: maximum (most severe).
      - Number of stories: mode (most common).
      - Annotation effort: maximum.
    """
    merged = {}
    for node in raw_nodes:
        b_id = node['building_id']
        if b_id not in merged:
            merged[b_id] = {
                'building_id': b_id,
                'global_coordinates': [node['global_coordinate']],
                'damage_states': [node['damage_state']],
                'num_stories_list': [node['num_stories']],
                'annotation_effort_list': [node['annotation_effort']],
                'frames': [node['frame']],
                'mask_filenames': [node['mask_filename']]
            }
        else:
            merged[b_id]['global_coordinates'].append(node['global_coordinate'])
            merged[b_id]['damage_states'].append(node['damage_state'])
            merged[b_id]['num_stories_list'].append(node['num_stories'])
            merged[b_id]['annotation_effort_list'].append(node['annotation_effort'])
            merged[b_id]['frames'].append(node['frame'])
            merged[b_id]['mask_filenames'].append(node['mask_filename'])
    final_nodes = []
    for b_id, data in merged.items():
        coords = np.array(data['global_coordinates'])
        avg_coord = coords.mean(axis=0).tolist()
        merged_damage = max(data['damage_states'])
        ns_mode = Counter(data['num_stories_list']).most_common(1)[0][0]
        ae_vals = [a for a in data['annotation_effort_list'] if a is not None]
        merged_effort = max(ae_vals) if ae_vals else None
        final_nodes.append({
            'building_id': b_id,
            'global_coordinate': avg_coord,
            'damage_state': merged_damage,
            'num_stories': ns_mode,
            'annotation_effort': merged_effort,
            'frame': data['frames'][0],   # representative frame
            'mask_filename': data['mask_filenames'][0]  # representative mask
        })
    return final_nodes

def get_building_patch(frame_path, mask_path, target_size=(224,224)):
    """
    Load a frame image and corresponding mask, compute the bounding box for the mask,
    crop the frame to that bounding box, and resize the patch.
    """
    try:
        frame_img = Image.open(frame_path).convert("RGB")
        mask_img = Image.open(mask_path).convert("L")
        mask_np = np.array(mask_img)
        coords = np.column_stack(np.where(mask_np > 0))
        if coords.size == 0:
            return frame_img.resize(target_size)
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0)
        cropped = frame_img.crop((x0, y0, x1, y1))
        return cropped.resize(target_size)
    except Exception as e:
        print(f"Error in cropping using {frame_path} and {mask_path}: {e}")
        return Image.open(frame_path).convert("RGB").resize(target_size)

def extract_visual_features(image, processor, model, pooling_mode="mean"):
    """
    Given a PIL image, preprocess it and run it through the transformer model.
    If pooling_mode is "cls", return the CLS token embedding.
    If pooling_mode is "mean", return the mean of the patch embeddings (discarding CLS token).
    Returns a torch.Tensor feature vector.
    """
    inputs = processor(image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    hidden_states = outputs.hidden_states[-1]  # shape: (1, seq_len, hidden_dim)
    if pooling_mode == "cls":
        feature = hidden_states[:, 0, :]
    elif pooling_mode == "mean":
        patch_tokens = hidden_states[:, 1:, :]
        feature = patch_tokens.mean(dim=1)
    else:
        raise ValueError("pooling_mode must be 'cls' or 'mean'")
    return feature.squeeze(0)

def build_node_features(nodes, frame_dir, mask_dir, processor, model, pooling_mode="mean"):
    """
    For each merged building node, load the corresponding frame and mask, extract the building patch,
    extract deep visual features, and concatenate them with metadata (num_stories and annotation_effort).
    Returns a tensor of node features and a tensor of ground truth damage states.
    """
    node_features = []
    ground_truth = []
    for node in nodes:
        frame_filename = os.path.basename(node['frame'])
        mask_filename = os.path.basename(node['mask_filename'])
        frame_path = os.path.join(frame_dir, frame_filename)
        mask_path = os.path.join(mask_dir, mask_filename)
        patch_img = get_building_patch(frame_path, mask_path)
        visual_feat = extract_visual_features(patch_img, processor, model, pooling_mode=pooling_mode)
        meta = torch.tensor([float(node['num_stories']),
                             float(node['annotation_effort']) if node['annotation_effort'] is not None else 0.0])
        node_feat = torch.cat([visual_feat, meta])
        node_features.append(node_feat)
        ground_truth.append(node['damage_state'])
    return torch.stack(node_features), torch.tensor(ground_truth)
