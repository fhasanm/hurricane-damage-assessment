import os
import random
import shutil

# Directories for original data
json_dir = "/home/fuad/Work/Projects/hurricane/PRJ-3278v2/DoriaNET/JSON"
frame_dir = "/home/fuad/Work/Projects/hurricane/PRJ-3278v2/DoriaNET/FRAME"
mask_dir = "/home/fuad/Work/Projects/hurricane/PRJ-3278v2/DoriaNET/MASK"

# Destination base directory for split data
dest_base = "/home/fuad/Work/Projects/hurricane/data"

# Define splits and create directories
splits = ['train', 'val', 'test']
for split in splits:
    os.makedirs(os.path.join(dest_base, split, "JSON"), exist_ok=True)
    os.makedirs(os.path.join(dest_base, split, "FRAME"), exist_ok=True)
    os.makedirs(os.path.join(dest_base, split, "MASK"), exist_ok=True)

# List JSON files (each JSON file corresponds to one frame)
json_files = sorted([f for f in os.listdir(json_dir) if f.endswith('.json')])
print(f"Found {len(json_files)} JSON files.")

# Shuffle JSON files for random splitting
random.shuffle(json_files)

# Define split ratios (e.g., train: 70%, val: 15%, test: 15%)
n = len(json_files)
n_train = int(0.7 * n)
n_val = int(0.15 * n)
n_test = n - n_train - n_val

train_files = json_files[:n_train]
val_files = json_files[n_train:n_train+n_val]
test_files = json_files[n_train+n_val:]

print(f"Train: {len(train_files)}, Validation: {len(val_files)}, Test: {len(test_files)}")

def copy_files(filenames, src_dirs, dest_dirs):
    """
    For each filename (JSON file), copy:
      - the JSON file from src_dirs["JSON"]
      - the corresponding frame image from src_dirs["FRAME"]
      - all mask files whose names start with the base frame id from src_dirs["MASK"]
    """
    for filename in filenames:
        base = os.path.splitext(filename)[0]  # e.g., "1_0373"
        
        # Copy JSON file
        src_json = os.path.join(src_dirs["JSON"], filename)
        dest_json = os.path.join(dest_dirs["JSON"], filename)
        shutil.copy(src_json, dest_json)
        
        # Copy corresponding frame image (assuming .jpg extension)
        frame_file = base + ".jpg"
        src_frame = os.path.join(src_dirs["FRAME"], frame_file)
        dest_frame = os.path.join(dest_dirs["FRAME"], frame_file)
        if os.path.exists(src_frame):
            shutil.copy(src_frame, dest_frame)
        else:
            print(f"Frame file {frame_file} not found.")
        
        # Copy corresponding mask files.
        # Assuming mask files start with the base name and an underscore.
        for f in os.listdir(src_dirs["MASK"]):
            if f.startswith(base + "_"):
                src_mask = os.path.join(src_dirs["MASK"], f)
                dest_mask = os.path.join(dest_dirs["MASK"], f)
                shutil.copy(src_mask, dest_mask)

# Define source directories
src_dirs = {
    "JSON": json_dir,
    "FRAME": frame_dir,
    "MASK": mask_dir
}

# Define destination directories for each split
dest_dirs_train = {
    "JSON": os.path.join(dest_base, "train", "JSON"),
    "FRAME": os.path.join(dest_base, "train", "FRAME"),
    "MASK": os.path.join(dest_base, "train", "MASK")
}
dest_dirs_val = {
    "JSON": os.path.join(dest_base, "val", "JSON"),
    "FRAME": os.path.join(dest_base, "val", "FRAME"),
    "MASK": os.path.join(dest_base, "val", "MASK")
}
dest_dirs_test = {
    "JSON": os.path.join(dest_base, "test", "JSON"),
    "FRAME": os.path.join(dest_base, "test", "FRAME"),
    "MASK": os.path.join(dest_base, "test", "MASK")
}

print("Copying train files...")
copy_files(train_files, src_dirs, dest_dirs_train)
print("Copying validation files...")
copy_files(val_files, src_dirs, dest_dirs_val)
print("Copying test files...")
copy_files(test_files, src_dirs, dest_dirs_test)

print("Data split complete.")
