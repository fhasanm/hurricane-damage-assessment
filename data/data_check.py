import os
import json
import numpy as np
from collections import Counter

def load_json_stats(split_dir):
    # split_dir should point to the folder containing the JSON files for that split
    json_dir = os.path.join(split_dir, "JSON")
    json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
    
    n_frames = len(json_files)
    building_counts = 0
    damage_states = []
    annotation_efforts = []
    num_stories = []
    
    for jf in json_files:
        json_path = os.path.join(json_dir, jf)
        with open(json_path, 'r') as f:
            data = json.load(f)
        buildings = data.get('Buildings', [])
        building_counts += len(buildings)
        for b in buildings:
            try:
                # Damage state: b[3], annotation effort: b[5] if available, number of stories: b[4]
                damage_states.append(int(b[3]))
                num_stories.append(int(b[4]))
                if len(b) > 5:
                    annotation_efforts.append(int(b[5]))
            except Exception as e:
                print(f"Error processing building in {jf}: {e}")
    
    stats = {
        "n_frames": n_frames,
        "n_buildings": building_counts,
        "damage_state_distribution": Counter(damage_states),
        "num_stories_distribution": Counter(num_stories),
        "annotation_effort_distribution": Counter(annotation_efforts)
    }
    return stats

# Paths to your split directories (adjust these paths if needed)
base_split_dir = "/home/fuad/Work/Projects/hurricane/data"
splits = ["train", "val", "test"]

for split in splits:
    split_dir = os.path.join(base_split_dir, split)
    stats = load_json_stats(split_dir)
    print(f"--- {split.upper()} ---")
    print(f"Frames: {stats['n_frames']}")
    print(f"Total building annotations: {stats['n_buildings']}")
    print("Damage state distribution:", stats["damage_state_distribution"])
    print("Number of stories distribution:", stats["num_stories_distribution"])
    print("Annotation effort distribution:", stats["annotation_effort_distribution"])
    print("\n")

