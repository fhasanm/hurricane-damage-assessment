import os
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
from skimage.exposure import match_histograms

# Directory containing drone frames
frames_dir = "/home/fuad/Work/Projects/hurricane/PRJ-3278v2/DoriaNET/FRAME"

# List image files (adjust extensions as needed)
image_files = sorted([f for f in os.listdir(frames_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
# Use the first 5 images
image_files = image_files[:5]

# Set the model name/path to the large classifier version.
model_name = "MITLL/LADI-v2-classifier-large-reference"

# Load the image processor and model with hidden states enabled.
processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModelForImageClassification.from_pretrained(model_name, output_hidden_states=True)

# Dictionaries to store each image's patch embeddings
image_patch_embeddings = {}
all_embeddings = []  # List to accumulate embeddings from all images

print("Extracting patch embeddings from images...")
for img_file in image_files:
    image_path = os.path.join(frames_dir, img_file)
    img = Image.open(image_path).convert("RGB")
    img_resized = img.resize((224, 224))
    
    inputs = processor(img_resized, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    
    hidden_states = outputs.hidden_states[-1]
    patch_embeddings = hidden_states[:, 1:, :]
    patch_embeddings_np = patch_embeddings.squeeze(0).cpu().numpy()
    image_patch_embeddings[img_file] = patch_embeddings_np
    all_embeddings.append(patch_embeddings_np)
    print(f"{img_file}: extracted {patch_embeddings_np.shape[0]} patches.")

all_embeddings_stacked = np.concatenate(all_embeddings, axis=0)
print(f"Total patches from all images: {all_embeddings_stacked.shape[0]}")

global_pca = PCA(n_components=3)
global_pca.fit(all_embeddings_stacked)

# Use percentile-based normalization
global_transformed = global_pca.transform(all_embeddings_stacked)
lower_percentile = np.percentile(global_transformed, 2)
upper_percentile = np.percentile(global_transformed, 98)
print(f"Global PCA percentiles: 2nd={lower_percentile}, 98th={upper_percentile}")

# We'll store the PCA-reduced grids for each image
pca_grids = {}

def process_and_transform(embeddings):
    n_patches = embeddings.shape[0]
    grid_rows = int(np.floor(np.sqrt(n_patches)))
    grid_cols = int(np.ceil(n_patches / grid_rows))
    total_slots = grid_rows * grid_cols
    if total_slots > n_patches:
        pad_amount = total_slots - n_patches
        padding = np.zeros((pad_amount, embeddings.shape[1]))
        flat_patches = np.concatenate([embeddings, padding], axis=0)
    else:
        flat_patches = embeddings
    patch_grid = flat_patches.reshape(grid_rows, grid_cols, -1)
    flat_patches_transformed = global_pca.transform(flat_patches)
    flat_patches_clipped = np.clip(flat_patches_transformed, lower_percentile, upper_percentile)
    flat_patches_normalized = (flat_patches_clipped - lower_percentile) / (upper_percentile - lower_percentile)
    patch_grid_normalized = flat_patches_normalized.reshape(grid_rows, grid_cols, 3)
    patch_grid_normalized = np.fliplr(patch_grid_normalized)  # Flip horizontally
    return patch_grid_normalized

# Process each image's patch embeddings and store the grid.
for img_file in image_files:
    embeddings = image_patch_embeddings[img_file]
    pca_grid = process_and_transform(embeddings)
    pca_grids[img_file] = pca_grid

# Use the PCA grid of the first image as reference for histogram matching.
ref_img_file = image_files[0]
ref_grid = pca_grids[ref_img_file]

# Create subplots: one row per image, two columns.
fig, axes = plt.subplots(len(image_files), 2, figsize=(12, 6 * len(image_files)))

for idx, img_file in enumerate(image_files):
    image_path = os.path.join(frames_dir, img_file)
    img = Image.open(image_path).convert("RGB")
    img_resized = img.resize((224, 224))
    
    grid = pca_grids[img_file]
    # For images other than the reference, apply histogram matching to the reference grid.
    if img_file != ref_img_file:
        grid_matched = match_histograms(grid, ref_grid, channel_axis=-1)
    else:
        grid_matched = grid
    
    axes[idx, 0].imshow(img_resized)
    axes[idx, 0].set_title(f"Original: {img_file}")
    axes[idx, 0].axis("off")
    
    axes[idx, 1].imshow(grid_matched)
    axes[idx, 1].set_title(f"PCA Features (Global, Histogram Matched) for {img_file}")
    axes[idx, 1].axis("off")

plt.tight_layout()
plt.show()
