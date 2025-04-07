# deep_features.py
import os
import json
import numpy as np
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from skimage.exposure import match_histograms

# ---------------------
# Data Loading and Merging
# ---------------------
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
    Merge nodes with the same building_id (i.e., the same building annotated in multiple frames).
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
    from collections import Counter
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
            # Use first mask & frame for reference; in practice, you can pick differently
            'mask_filename': data['mask_filenames'][0],
            'frame': data['frames'][0]
        })
    return final_nodes

# ---------------------
# Building-Patch Extraction
# ---------------------
def get_building_patch(frame_path, mask_path, target_size=(224,224)):
    """
    Load a frame image and a corresponding mask, compute the bounding box of the mask,
    crop the frame to that bounding box, and resize the patch.
    """
    try:
        frame_img = Image.open(frame_path).convert("RGB")
        mask_img = Image.open(mask_path).convert("L")
        mask_np = np.array(mask_img)
        coords = np.column_stack(np.where(mask_np > 0))
        if coords.size == 0:
            # If mask is empty or invalid, return a resized entire frame
            return frame_img.resize(target_size)
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0)
        cropped = frame_img.crop((x0, y0, x1, y1))
        return cropped.resize(target_size)
    except Exception as e:
        print(f"Error in cropping using {frame_path} and {mask_path}: {e}")
        # Fallback: load the entire frame resized
        return Image.open(frame_path).convert("RGB").resize(target_size)

# ---------------------
# Visual Feature Extraction
# ---------------------
def extract_visual_features(
    image,
    processor,
    model,
    pooling_mode="mean",
    fine_tune_transformer=False
):
    """
    Given a PIL image, preprocess it and run it through the transformer model.
    If pooling_mode is "cls", return the CLS token embedding.
    If pooling_mode is "mean", return the mean of the patch embeddings (excluding the CLS token).
    
    If fine_tune_transformer is True, we do not wrap the forward pass in `torch.no_grad()`,
    so the transformer's weights can be updated by backprop. If it's False, we freeze them.
    """
    inputs = processor(image, return_tensors="pt")
    
    if fine_tune_transformer:
        # Gradients enabled
        outputs = model(**inputs)
    else:
        # No gradients for the transformer => frozen
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

# ---------------------
# Node Feature Building
# ---------------------
def build_node_features(
    nodes,
    frame_dir,
    mask_dir,
    processor,
    model,
    pooling_mode="mean",
    fine_tune_transformer=False
):
    """
    For each merged building node, load the corresponding frame and mask,
    extract the building patch, extract visual features, and concatenate them
    with metadata (num_stories and annotation_effort).
    Returns a tensor of node features and a tensor of ground truth damage states.
    
    The `fine_tune_transformer` flag controls whether we freeze or unfreeze the transformer.
    """
    from os.path import basename, join
    node_features = []
    ground_truth = []

    for node in nodes:
        frame_path = join(frame_dir, basename(node['frame']))
        mask_path = join(mask_dir, basename(node['mask_filename']))
        patch_img = get_building_patch(frame_path, mask_path)
        
        visual_feat = extract_visual_features(
            patch_img,
            processor,
            model,
            pooling_mode=pooling_mode,
            fine_tune_transformer=fine_tune_transformer
        )
        
        # Make sure visual_feat is 1D (e.g., shape [hidden_dim])
        if visual_feat.dim() > 1:
            # Flatten or squeeze the 0th dimension if it's [1, hidden_dim]
            visual_feat = visual_feat.view(-1)
        
        annotation_effort = node['annotation_effort'] if node['annotation_effort'] is not None else 0.0
        meta = torch.tensor([float(node['num_stories']), float(annotation_effort)])
        # Also ensure meta is 1D
        meta = meta.view(-1)
        
        # Now both visual_feat and meta are 1D => safe to concatenate
        node_feat = torch.cat([visual_feat, meta], dim=0)
        
        node_features.append(node_feat)
        ground_truth.append(node['damage_state'])
    
    return torch.stack(node_features), torch.tensor(ground_truth)


# ---------------------
# Collage Function for Building Patches
# ---------------------
def plot_deep_features_collage_for_buildings(
    nodes,
    frame_dir,
    mask_dir,
    processor,
    model,
    pooling_mode="mean",
    n_examples=4,
    plot=True,
    fine_tune_transformer=False
):
    """
    Randomly select n_examples buildings from the given node list and:
      - Extract the patch for each building (top row).
      - Perform PCA on the patch's token embeddings (excluding CLS) and visualize them (bottom row).
    
    If fine_tune_transformer=True, do not wrap the forward pass in torch.no_grad(),
    so the transformer's weights could be updated. In practice, for static collage,
    you likely won't be backpropagating, but the logic is consistent with the build_node_features usage.
    """
    if not plot or len(nodes) == 0:
        return
    
    import random
    selected_nodes = random.sample(nodes, min(n_examples, len(nodes)))

    orig_imgs = {}
    embeddings_list = []
    node_ids = []
    
    print("Extracting patch embeddings for building collage...")
    for node in selected_nodes:
        b_id = node['building_id']
        frame_path = os.path.join(frame_dir, os.path.basename(node['frame']))
        mask_path = os.path.join(mask_dir, os.path.basename(node['mask_filename']))

        # Building patch
        patch_img = get_building_patch(frame_path, mask_path)
        orig_imgs[b_id] = patch_img

        # Extract token embeddings
        inputs = processor(patch_img, return_tensors="pt")
        if fine_tune_transformer:
            # Gradients on
            outputs = model(**inputs)
        else:
            # Frozen
            with torch.no_grad():
                outputs = model(**inputs)
        
        hidden_states = outputs.hidden_states[-1]  # shape [1, seq_len, hidden_dim]
        patch_tokens = hidden_states[:, 1:, :]     # exclude CLS
        emb_np = patch_tokens.squeeze(0).cpu().numpy()

        embeddings_list.append(emb_np)
        node_ids.append(b_id)
        print(f"Building {b_id}: extracted {emb_np.shape[0]} patch tokens.")

    if len(embeddings_list) == 0:
        print("No embeddings found. Aborting collage.")
        return
    
    all_embeddings_stacked = np.concatenate(embeddings_list, axis=0)
    print(f"Total patch tokens from selected buildings: {all_embeddings_stacked.shape[0]}")

    pca = PCA(n_components=3)
    pca.fit(all_embeddings_stacked)

    transformed_all = pca.transform(all_embeddings_stacked)
    lower = np.percentile(transformed_all, 2)
    upper = np.percentile(transformed_all, 98)
    print(f"Global PCA percentiles: 2nd={lower}, 98th={upper}")

    def process_and_transform(embeddings):
        """
        Project embeddings to 3D via PCA, clip to [lower, upper], then normalize to [0,1].
        Finally, reshape into a grid (~ sqrt(num_tokens) x sqrt(num_tokens)).
        """
        n_patches = embeddings.shape[0]
        grid_rows = int(np.floor(np.sqrt(n_patches)))
        grid_cols = int(np.ceil(n_patches / grid_rows))
        total_slots = grid_rows * grid_cols
        if total_slots > n_patches:
            pad = np.zeros((total_slots - n_patches, embeddings.shape[1]))
            flat = np.concatenate([embeddings, pad], axis=0)
        else:
            flat = embeddings

        transformed = pca.transform(flat)
        clipped = np.clip(transformed, lower, upper)
        normalized = (clipped - lower) / (upper - lower)
        grid = normalized.reshape(grid_rows, grid_cols, 3)
        return grid

    from skimage.exposure import match_histograms
    building_grids = {}
    offset = 0
    for i, node_id in enumerate(node_ids):
        emb_current = embeddings_list[i]
        count = emb_current.shape[0]
        subset = all_embeddings_stacked[offset : offset + count]
        offset += count
        grid = process_and_transform(subset)
        building_grids[node_id] = grid

    # Use the first building's PCA grid as the reference for histogram matching
    ref_id = node_ids[0]
    ref_grid = building_grids[ref_id]
    matched_grids = {}
    for node_id in node_ids:
        grid = building_grids[node_id]
        if node_id != ref_id:
            grid_matched = match_histograms(grid, ref_grid, channel_axis=-1)
        else:
            grid_matched = grid
        matched_grids[node_id] = grid_matched

    fig, axes = plt.subplots(2, len(selected_nodes), figsize=(4 * len(selected_nodes), 8))
    for idx, node_id in enumerate(node_ids):
        # Top row: building patch
        axes[0, idx].imshow(orig_imgs[node_id])
        axes[0, idx].set_title(f"Patch: bldg {node_id}")
        axes[0, idx].axis("off")

        # Bottom row: PCA grid
        axes[1, idx].imshow(matched_grids[node_id])
        axes[1, idx].set_title(f"PCA Features: bldg {node_id}")
        axes[1, idx].axis("off")

    plt.tight_layout()
    plt.show()
