import os
import json
import numpy as np
from sklearn.neighbors import NearestNeighbors
from pyvis.network import Network
from collections import Counter

# Directory containing JSON annotations
json_dir = "/home/fuad/Work/Projects/hurricane/PRJ-3278v2/DoriaNET/JSON"  # JSON annotations folder

# List JSON files in the JSON folder
json_files = sorted([f for f in os.listdir(json_dir) if f.endswith('.json')])
print(f"Found {len(json_files)} JSON files.")

# Initialize a list for building nodes (each node from an annotation)
raw_nodes = []  
node_id = 0

# Iterate over JSON files and extract building information.
for jf in json_files:
    json_path = os.path.join(json_dir, jf)
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    frame_name = data.get('Frame_Name', None)
    buildings = data.get('Buildings', [])
    for b in buildings:
        # Expected format for each building:
        # [building_id, global_coordinate, mask_filename, damage_state, num_stories, annotation_effort]
        try:
            # Convert global coordinate (assumed to be a string representing a matrix)
            coord = np.array(np.matrix(b[1])).flatten()  # e.g., [latitude, longitude]
            if coord.size < 2:
                continue
        except Exception as e:
            print(f"Error parsing coordinate for building {b[0]} in file {jf}: {e}")
            continue
        
        raw_nodes.append({
            'node_id': int(node_id),
            'building_id': b[0],
            'global_coordinate': [float(coord[0]), float(coord[1])],
            'mask_filename': b[2],
            'damage_state': int(b[3]),
            'num_stories': int(b[4]),
            'annotation_effort': int(b[5]) if len(b) > 5 else None,
            'frame': frame_name
        })
        node_id += 1

print(f"Total raw nodes loaded: {len(raw_nodes)}")

# Merge nodes with the same building_id.
merged_nodes = {}
for node in raw_nodes:
    b_id = node['building_id']
    if b_id not in merged_nodes:
        merged_nodes[b_id] = {
            'building_id': b_id,
            'global_coordinates': [node['global_coordinate']],
            'damage_states': [node['damage_state']],
            'num_stories_list': [node['num_stories']],
            'annotation_effort_list': [node['annotation_effort']],
            'frames': [node['frame']]
        }
    else:
        merged_nodes[b_id]['global_coordinates'].append(node['global_coordinate'])
        merged_nodes[b_id]['damage_states'].append(node['damage_state'])
        merged_nodes[b_id]['num_stories_list'].append(node['num_stories'])
        merged_nodes[b_id]['annotation_effort_list'].append(node['annotation_effort'])
        merged_nodes[b_id]['frames'].append(node['frame'])

# Create a new list of nodes from merged data.
final_nodes = []
new_node_id = 0
for b_id, data in merged_nodes.items():
    # Average the global coordinates
    coords = np.array(data['global_coordinates'])
    avg_coord = coords.mean(axis=0).tolist()
    
    # For damage state, take the maximum (i.e. most severe)
    merged_damage = max(data['damage_states'])
    
    # For number of stories, take the mode (or first element if you assume they are the same)
    ns_counter = Counter(data['num_stories_list'])
    merged_stories = ns_counter.most_common(1)[0][0]
    
    # For annotation effort, take the maximum (i.e. most difficult)
    ae = data['annotation_effort_list']
    merged_effort = max([a for a in ae if a is not None]) if any(a is not None for a in ae) else None
    
    final_nodes.append({
        'node_id': new_node_id,
        'building_id': b_id,
        'global_coordinate': avg_coord,
        'damage_state': merged_damage,
        'num_stories': merged_stories,
        'annotation_effort': merged_effort,
        'frames': data['frames']
    })
    new_node_id += 1

print(f"Total merged nodes (unique buildings): {len(final_nodes)}")

# Build an array of global coordinates (latitude, longitude) for all final nodes.
coords_final = np.array([node['global_coordinate'] for node in final_nodes])  # shape: (N, 2)

# Construct graph edges based on spatial proximity using k-nearest neighbors.
k = 2  # number of neighbors to connect
nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(coords_final)
distances, indices = nbrs.kneighbors(coords_final)

# Create a PyVis network.
net = Network(height='750px', width='100%', notebook=False)

# Define a color map based on damage state.
color_map = {
    0: 'green',    # No damage or very minor damage
    1: 'blue',     # Minor damage
    2: 'yellow',   # Moderate damage
    3: 'orange',   # Severe damage
    4: 'red',      # Destruction
    5: 'purple'    # Under construction/destroyed
}

# Add nodes to the PyVis network.
for node in final_nodes:
    n_id = int(node['node_id'])
    # Use longitude for x and latitude for y.
    lon = float(node['global_coordinate'][1])
    lat = float(node['global_coordinate'][0])
    damage = node['damage_state']
    color = color_map.get(damage, 'gray')
    label = f"ID: {node['building_id']}\nDamage: {damage}\nStories: {node['num_stories']}"
    net.add_node(n_id, label=label, title=label, color=color, x=lon, y=lat)

# Add edges (skipping self-loops).
for i, nbr_indices in enumerate(indices):
    for j in nbr_indices[1:]:
        net.add_edge(int(i), int(j))

# Disable physics for a static graph.
net.set_options("""
var options = {
  "physics": {
    "enabled": false
  }
}
""")

net.show("dorianet_graph.html", notebook=False)
print("Graph visualization saved as 'dorianet_graph.html'. Open this file in your browser to interact with it.")
