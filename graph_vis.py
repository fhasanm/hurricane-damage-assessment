

import os
import json
import numpy as np
from sklearn.neighbors import NearestNeighbors
from pyvis.network import Network

# Directory containing JSON annotations
json_dir = "/home/fuad/Work/Projects/hurricane/PRJ-3278v2/DoriaNET/JSON"  # adjust path if needed

# List JSON files in the JSON folder
json_files = sorted([f for f in os.listdir(json_dir) if f.endswith('.json')])
print(f"Found {len(json_files)} JSON files.")

# Initialize a list for building nodes.
nodes = []  # Each node is a dictionary with building info.
node_id = 0

# Iterate over JSON files and extract building information.
for jf in json_files:
    json_path = os.path.join(json_dir, jf)
    with open(json_path, 'r') as f:
        data = json.load(f)

    frame_name = data.get('Frame_Name', None)
    buildings = data.get('Buildings', [])
    for b in buildings:
        try:
            # Convert global coordinate string to a NumPy array.
            coord = np.array(np.matrix(b[1])).flatten()  # e.g., [latitude, longitude]
            if coord.size < 2:
                continue
        except Exception as e:
            print(f"Error parsing coordinate for building {b[0]} in {jf}: {e}")
            continue

        nodes.append({
            'node_id': int(node_id),
            'building_id': b[0],
            'global_coordinate': [float(coord[0]), float(coord[1])],  # converting to Python float
            'damage_state': int(b[3]) if isinstance(b[3], (int, np.integer)) else b[3],
            'num_stories': int(b[4]) if isinstance(b[4], (int, np.integer)) else b[4],
            'annotation_effort': int(b[5]) if len(b) > 5 and isinstance(b[5], (int, np.integer)) else (b[5] if len(b) > 5 else None),
            'frame': frame_name
        })
        node_id += 1

print(f"Total buildings (nodes) loaded: {len(nodes)}")

# Create an array of global coordinates (latitude, longitude) for all nodes.
coords = np.array([node['global_coordinate'] for node in nodes])  # shape: (N, 2)

# Build graph edges based on spatial proximity using k-nearest neighbors.
k = 4  # number of neighbors to connect
nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(coords)
distances, indices = nbrs.kneighbors(coords)

# Create a PyVis network.
net = Network(height='750px', width='100%', notebook=False)

# Define a color map based on damage state.
color_map = {
    0: 'purple',    # No damage or very minor damage
    1: 'blue',     # Minor damage
    2: 'yellow',   # Moderate damage
    3: 'orange',   # Severe damage
    4: 'red',      # Destruction
    5: 'black'    # Under construction/destroyed
}

# Add nodes to the PyVis network.
# Convert NumPy types to native Python types.
for node in nodes:
    n_id = int(node['node_id'])
    lon = float(node['global_coordinate'][1])  # x-axis
    lat = float(node['global_coordinate'][0])    # y-axis
    damage = node['damage_state']
    color = color_map.get(damage, 'gray')
    label = f"ID: {node['building_id']}\nDamage: {damage}\nStories: {node['num_stories']}"
    net.add_node(n_id, label=label, title=label, color=color, x=lon, y=lat)

# Add edges (skipping self-loops)
for i, neighbors in enumerate(indices):
    for j in neighbors[1:]:
        net.add_edge(int(i), int(j))


# Enable physics (optional) and show the interactive graph.
net.toggle_physics(True)
net.show("dorianet_graph.html", notebook=False)
print("Graph visualization saved as 'dorianet_graph.html'. Open this file in a browser to interact with it.")



