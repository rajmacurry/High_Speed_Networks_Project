import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy as sp

city = "Norden, Bremen, Hamburg, Hannover, Berlin, Leipzig, Nurnberg, Munich, Ulm, Stuttgart, Karlsruhe, Frankfurt, Koln, Dusseldorf, Essen, Dortmund"

N = 16

city_list = [c.strip() for c in city.split(',')]
G = nx.Graph()
G.add_nodes_from(city_list)

# Define physical links (edges) based on a typical German backbone topology
edges = [
    ("Norden", "Bremen"), ("Norden","Dortmund"), ("Bremen", "Hamburg"), ("Hamburg", "Hannover"),
    ("Hannover", "Berlin"), ("Hannover", "Bremen"), ("Berlin", "Leipzig"), ("Leipzig", "Nurnberg"), ("Nurnberg", "Munich"), 
    ("Munich", "Ulm"), ("Ulm", "Stuttgart"), ("Stuttgart", "Karlsruhe"), ("Frankfurt", "Koln"), 
    ("Koln", "Dusseldorf"), ("Dusseldorf", "Essen"), ("Essen", "Dortmund"), ("Dortmund", "Hannover"),
    ("Frankfurt", "Hannover"), ("Koln", "Dortmund"), ("Hamburg", "Berlin"), ("Hannover", "Leipzig"), 
    ("Leipzig", "Frankfurt"), ("Frankfurt", "Nurnberg"), ("Stuttgart","Nurnberg"), ("Frankfurt", "Karlsruhe")
]
G.add_edges_from(edges)

# Create the adjacency matrix
adj_matrix = nx.to_numpy_array(G)
df_topology = pd.DataFrame(adj_matrix, index=G.nodes(), columns=G.nodes())

# Calculate degree statistics
degrees = df_topology.sum(axis=1)
print("Degrees of each node:")
print(degrees)

min_degree = degrees.min()
max_degree = degrees.max()
avg_degree = degrees.mean()
var_degree = degrees.var(ddof=0)  # population variance

print(f"Minimum degree: {min_degree}")
print(f"Maximum degree: {max_degree}")
print(f"Average degree: {avg_degree:.2f}")
print(f"Variance of degree: {var_degree:.2f}")

# Plot degree distribution histogram
bins = np.arange(min_degree - 0.5, max_degree + 1.5, 1)

degrees.plot(kind='hist', 
             bins=bins,
             edgecolor='black',
             rwidth=0.8,
             figsize=(10, 6))
plt.xlabel('Node Degree')
plt.ylabel('Number of Nodes (Frequency)')
plt.title('Node Degree Distribution Histogram')
plt.grid(axis='y', alpha=0.5)
plt.show()

# Create weighted matrix
weighted_matrix = df_topology.copy()

# Define weighted edges
weighted_edges = {
    ("Norden", "Bremen"): 121.73,
    ("Norden", "Dortmund"): 232.84,
    ("Bremen", "Hamburg"): 94.8,
    ("Hamburg", "Hannover"): 131.22,
    ("Hamburg", "Berlin"): 256.28,
    ("Hannover", "Berlin"): 250.91,
    ("Hannover", "Bremen"): 101,
    ("Hannover", "Leipzig"): 215.78,
    ("Hannover", "Frankfurt"): 262.53,
    ("Hannover", "Dortmund"): 183.11,
    ("Berlin", "Leipzig"): 149.91,
    ("Leipzig", "Nurnberg"): 230.08,
    ("Leipzig", "Frankfurt"): 294.3,
    ("Nurnberg", "Munich"): 151.51,
    ("Nurnberg", "Stuttgart"): 157.87,
    ("Nurnberg", "Frankfurt"): 186.99,
    ("Munich", "Ulm"): 122.06,
    ("Stuttgart", "Karlsruhe"): 62.93,
    ("Karlsruhe", "Frankfurt"): 124.12,
    ("Frankfurt", "Koln"): 152.52,
    ("Koln", "Dusseldorf"): 34.25,
    ("Koln", "Dortmund"): 73.17,
    ("Dusseldorf", "Essen"): 31.56,
    ("Essen", "Dortmund"): 31.63
}

for (src, dst), weight in weighted_edges.items():
    weighted_matrix.loc[src, dst] = weight
    weighted_matrix.loc[dst, src] = weight

# Scale the weighted matrix
weighted_matrix = weighted_matrix * 1.8
print("Weighted matrix (scaled by 1.8):")
print(weighted_matrix)

# Import routing module (assuming it exists)
try:
    import routing_v2_proj as routing
    from routing_v2_proj import Graph
    
    weighted_list = weighted_matrix.values.tolist()
    unweighted_list = df_topology.values.tolist()
    print(f"Weighted matrix shape: {len(weighted_list)}x{len(weighted_list[0])}")
    print(f"Unweighted matrix shape: {len(unweighted_list)}x{len(unweighted_list[0])}")
    
    # For UNWEIGHTED Graph
    print("\n=== UNWEIGHTED GRAPH ===")
    graph = routing.Graph()
    unweighted_paths = routing.shortestPaths(graph, unweighted_list)
    hop_matrix_unweighted = routing.countHops(unweighted_paths)
    print(f"Number of nodes: {len(unweighted_list)}")
    print(f"Hop matrix (sample): {hop_matrix_unweighted[:3]}")
    
    # For WEIGHTED Graph
    print("\n=== WEIGHTED GRAPH ===")
    weighted_paths = routing.shortestPaths(graph, weighted_list)
    hop_matrix_weighted = routing.countHops(weighted_paths)
    
    def extract_distances_and_paths(paths_list):
        """Extract distances and paths for all node pairs"""
        all_distances = []
        
        for src_paths in paths_list:
            for p in src_paths:
                all_distances.append({
                    'source': p['source'],
                    'destination': p['destination'],
                    'distance': p['distance'],
                    'path': p['path'][0] if p['path'] else []
                })
        
        return all_distances
    
    weighted_distances = extract_distances_and_paths(weighted_paths)
    unweighted_distances = extract_distances_and_paths(unweighted_paths)
    
    print(f"Sample weighted path: {weighted_distances[38]}")
    
    # Extract hops and distances
    hops_data = []
    distances_data = []
    
    for item in weighted_distances:
        if item["source"] != item["destination"]:
            hops = len(item["path"]) - 1
            hops_data.append(hops)
            distances_data.append(item["distance"])
    
    # Histogram: Number of Hops
    plt.figure(figsize=(10, 6))
    bins_hops = np.arange(min(hops_data) - 0.5, max(hops_data) + 1.5, 1)
    plt.hist(hops_data, bins=bins_hops, edgecolor="black", rwidth=0.8)
    plt.xlabel("Number of Hops")
    plt.ylabel("Frequency")
    plt.title("Histogram of Number of Hops")
    plt.grid(axis="y", alpha=0.5)
    plt.show()
    
    # Histogram: Distances
    plt.figure(figsize=(10, 6))
    plt.hist(distances_data, bins=25, edgecolor="black", rwidth=0.8)
    plt.xlabel("Distance")
    plt.ylabel("Frequency")
    plt.title("Histogram of Node Distances")
    plt.grid(axis="y", alpha=0.5)
    plt.show()
    
    # Calculate average hops per demand
    N_bi = N * (N - 1) / 2
    sum_h = int(np.triu(np.array(hop_matrix_unweighted), k=1).sum())
    avg_no_hops_per_demand = sum_h / N_bi
    
    print(f"Total hops (upper triangle): {sum_h}")
    print(f"Average hops per demand: {avg_no_hops_per_demand:.2f}")
    
    # Semi-empirical formulas
    semi_emp_avg_no_hops_per_demand_1 = np.sqrt((N - 2) / (avg_degree - 1))
    semi_emp_avg_no_hops_per_demand_2 = 1.12 * np.sqrt(N / avg_degree)
    
    print(f"Semi-empirical formula 1: {semi_emp_avg_no_hops_per_demand_1:.2f}")
    print(f"Semi-empirical formula 2: {semi_emp_avg_no_hops_per_demand_2:.2f}")
    
except ImportError:
    print("Routing module not found. Skipping routing analysis.")

# Create logical topology graph
H = np.array(hop_matrix_unweighted) if 'hop_matrix_unweighted' in locals() else np.zeros((N, N))
G_logical = nx.Graph()
G_logical.add_nodes_from(city_list)

for i in range(N):
    for j in range(i + 1, N):
        G_logical.add_edge(
            city_list[i],
            city_list[j],
            hops=int(H[i, j])
        )

plt.figure(figsize=(8,6))
pos = nx.spring_layout(G_logical, seed=42)
nx.draw(G_logical, pos, with_labels=True, node_size=900, font_size=8, alpha=0.6)
edge_labels = nx.get_edge_attributes(G_logical, "hops")
nx.draw_networkx_edge_labels(G_logical, pos, edge_labels=edge_labels, font_size=6)
plt.title("Full Logical Topology (Hop Distances)")
plt.show()

# Create logical hop tree from Frankfurt
root = "Frankfurt"
tree = nx.single_source_shortest_path_length(G, root)
G_tree = nx.Graph()
for dst, hops in tree.items():
    if dst != root:
        G_tree.add_edge(root, dst, hops=hops)

plt.figure(figsize=(10, 8))
pos = nx.spring_layout(G_tree, seed=42)
nx.draw(G_tree, pos, with_labels=True, node_size=1200)
nx.draw_networkx_edge_labels(G_tree, pos, edge_labels=nx.get_edge_attributes(G_tree, "hops"))
plt.title(f"Logical Hop Tree from {root}")
plt.show()

# Connectivity analysis
min_degree = min(dict(G.degree()).values())
node_connectivity = nx.node_connectivity(G)
edge_connectivity = nx.edge_connectivity(G)
algebraic_connectivity = nx.algebraic_connectivity(G)

print(f"Minimum node degree δ(G): {min_degree}")
print(f"Node connectivity κ(G): {node_connectivity}")
print(f"Edge connectivity λ(G): {edge_connectivity}")
print(f"Algebraic connectivity μ₂: {algebraic_connectivity:.4f}")

# Cutset analysis
X = "Hamburg"
Y = "Stuttgart"

# Minimum x–y edge cut
edge_cut = nx.minimum_edge_cut(G, X, Y)
print("Edge cutset:")
print(edge_cut)
print("Cut size:", len(edge_cut))

# Minimum x–y node cut
node_cut = nx.minimum_node_cut(G, X, Y)
print("Node cutset:")
print(node_cut)
print("Cut size:", len(node_cut))

# Path analysis for unweighted graph
print("\n=== UNWEIGHTED GRAPH PATH ANALYSIS ===")
x = "Hamburg"
y = "Stuttgart"

# Service (primary) path: minimum-hop path
service_path = nx.shortest_path(G, x, y)
print("Service path:")
print(service_path)
print("Hops:", len(service_path) - 1)

# All simple paths between x and y
all_paths = list(nx.all_simple_paths(G, x, y))
print(f"Total simple paths: {len(all_paths)}")

# Edge-disjoint backup paths
print("\n1) Mutually edge-disjoint backup paths:")
service_path = nx.shortest_path(G, x, y)
used_edges = set(frozenset((u, v)) for u, v in zip(service_path[:-1], service_path[1:]))
all_paths_sorted = sorted(all_paths, key=lambda p: len(p) - 1)
edge_disjoint_backups = []

for path in all_paths_sorted:
    path_edges = set(frozenset((u, v)) for u, v in zip(path[:-1], path[1:]))
    if path_edges.isdisjoint(used_edges):
        edge_disjoint_backups.append(path)
        used_edges.update(path_edges)
    if len(edge_disjoint_backups) == 3:
        break

for p in edge_disjoint_backups:
    print(p, "hops:", len(p) - 1)

# Node-disjoint backup paths
print("\n2) Mutually node-disjoint backup paths:")
service_path = nx.shortest_path(G, x, y)
used_nodes = set(service_path[1:-1])
all_paths_sorted = sorted(all_paths, key=lambda p: len(p) - 1)
node_disjoint_backups = []

for path in all_paths_sorted:
    path_nodes = set(path[1:-1])
    if path_nodes.isdisjoint(used_nodes):
        node_disjoint_backups.append(path)
        used_nodes.update(path_nodes)
    if len(node_disjoint_backups) == 3:
        break

for p in node_disjoint_backups:
    print(p, "hops:", len(p) - 1)

# Path analysis for weighted graph
print("\n=== WEIGHTED GRAPH PATH ANALYSIS ===")
# Build weighted graph
Gw = nx.Graph()
for i in weighted_matrix.index:
    for j in weighted_matrix.columns:
        w = weighted_matrix.loc[i, j]
        if w > 0:
            Gw.add_edge(i, j, weight=w)

# Weighted service path
service_path = nx.shortest_path(Gw, x, y, weight="weight")
service_dist = nx.shortest_path_length(Gw, x, y, weight="weight")
print("Weighted service path:")
print(service_path, "distance:", service_dist)

# Edge-disjoint backup paths (weighted)
print("\n1) Weighted mutually edge-disjoint backup paths:")
used_edges = set(frozenset((u, v)) for u, v in zip(service_path[:-1], service_path[1:]))
candidate_paths = list(nx.shortest_simple_paths(Gw, x, y, weight="weight"))
edge_disjoint_backups = []

for path in candidate_paths:
    path_edges = set(frozenset((u, v)) for u, v in zip(path[:-1], path[1:]))
    if path_edges.isdisjoint(used_edges):
        edge_disjoint_backups.append(path)
        used_edges.update(path_edges)
    if len(edge_disjoint_backups) == 3:
        break

for p in edge_disjoint_backups:
    dist = nx.path_weight(Gw, p, weight="weight")
    print(p, "distance:", dist)

# Node-disjoint backup paths (weighted)
print("\n2) Weighted mutually node-disjoint backup paths:")
service_path = nx.shortest_path(Gw, x, y, weight="weight")
used_nodes = set(service_path[1:-1])
candidate_paths = list(nx.shortest_simple_paths(Gw, x, y, weight="weight"))
node_disjoint_backups = []

for path in candidate_paths:
    path_nodes = set(path[1:-1])
    if path_nodes.isdisjoint(used_nodes):
        node_disjoint_backups.append(path)
        used_nodes.update(path_nodes)
    if len(node_disjoint_backups) == 3:
        break

for p in node_disjoint_backups:
    dist = nx.path_weight(Gw, p, weight="weight")
    print(p, "distance:", dist)

# Traffic matrix analysis
print("\n=== TRAFFIC MATRIX ANALYSIS ===")
X = 28
Y = 47
Z = 53

symbol_map = {
    "X": X,
    "Y": Y,
    "Z": Z,
    "0": 0
}

traffic_symbols = [
    ["0","X","Y","Z","X","Y","0","X","Y","0","X","0","0","0","0","0"],
    ["X","0","X","Y","0","X","0","Z","X","Y","Z","0","0","Z","0","Z"],
    ["Y","X","0","X","Y","0","X","0","Z","X","0","Z","0","0","0","0"],
    ["Z","Y","X","0","X","Y","Z","0","0","Z","X","0","Z","0","0","0"],
    ["X","0","Y","X","0","X","Y","Z","0","Y","0","0","0","Z","0","Z"],
    ["Y","X","0","Y","X","0","X","Y","Z","X","Y","0","X","0","X","0"],
    ["0","0","X","Z","Y","X","0","X","Y","Z","X","0","Z","X","0","X"],
    ["X","Z","0","0","Z","Y","X","0","X","Y","Z","X","0","Z","0","Z"],
    ["0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0"],
    ["Y","X","Z","0","0","Z","Y","X","0","X","Y","Z","X","Y","X","Y"],
    ["0","Y","X","Z","Y","X","Z","Y","X","0","X","Y","Z","X","Z","X"],
    ["X","Z","0","X","0","Y","X","Z","Y","X","0","X","Y","Z","X","Z"],
    ["0","0","Z","0","0","0","0","X","Z","Y","X","0","X","Y","X","Y"],
    ["0","0","0","Z","0","X","Z","0","X","Z","Y","X","0","X","0","X"],
    ["0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0"],
    ["0","Z","0","0","Z","0","X","Z","Y","X","Z","Y","X","0","X","0"]
]

traffic_matrix = np.array([
    [symbol_map[v] for v in row]
    for row in traffic_symbols
])

df_traffic = pd.DataFrame(
    traffic_matrix,
    index=city_list,
    columns=city_list
)

print("Traffic matrix:")
print(df_traffic)

# Zero out Essen and Ulm nodes
zero_nodes = ["Essen", "Ulm"]
zero_indices = [city_list.index(n) for n in zero_nodes]
print(f"\nZero indices for {zero_nodes}: {zero_indices}")

for idx in zero_indices:
    traffic_matrix[idx, :] = 0
    traffic_matrix[:, idx] = 0


# Create demand matrix (binary)
demand_matrix = (traffic_matrix > 0).astype(int)
print("\nDemand matrix (binary):")
print(pd.DataFrame(demand_matrix, index=city_list, columns=city_list))

# ---------------------------------------------------------
# ROUTING ANALYSIS
# ---------------------------------------------------------
print("\n=== ROUTING ANALYSIS ===")

strategies = ["shortest", "longest", "largest"]
results = {}

# Ensure MAX_LINK_CAP is uncapacitated
routing.MAX_LINK_CAP = 999999

# Convert numpy arrays to lists for the routing module
adj_list = df_topology.values.tolist()
traffic_list = traffic_matrix.tolist()

# Get initial Shortest Paths (Dijkstra)
# Note: Graph object needed
graph_obj = routing.Graph()
initial_paths = routing.shortestPaths(graph_obj, adj_list)
hop_matrix = routing.create_traffic_matrix(adj_list, None) # This seems to be used for hops in original code, let's check
# Actually, countHops returns the hop matrix
hop_matrix_raw, _ = routing.hop_count(initial_paths) # countHops in original code returns matrix, but signature seems to be creating it from paths?
# Let's check routing_v2_proj.py def countHops(paths: list): ... return hops_matrix
# The original code uses: hop_matrix = countHops(paths)
hop_matrix_initial = routing.countHops(initial_paths)

link_loads_data = {}

for strategy in strategies:
    print(f"\n--- Strategy: {strategy} ---")
    
    # 1. Order Paths
    # Note: hop_matrix_initial is list of lists
    ordered_ids = routing.orderPaths(initial_paths, traffic_list, hop_matrix_initial, order=strategy)
    
    # 2. Route
    # route(ordered_traffic_demands, load_matrix, matrix)
    # Returns: load_matrix, path_matrix, distance_matrix, blocked_traffic, blocked_paths
    
    # Initialize empty load matrix
    current_load_matrix_input = routing.create_load_matrix(adj_list)
    
    # Run routing
    # We pass a COPY of adj_list because route might modify it (it calls update_network)
    # But for uncapacitated, it shouldn't matter much, but safe is better.
    final_load_matrix, final_path_matrix, final_dist_matrix, blocked_traffic, blocked_paths = routing.route(
        ordered_ids, 
        current_load_matrix_input, 
        [row[:] for row in adj_list] # deep copy of matrix
    )
    
    print(f"Blocked Traffic: {blocked_traffic}")
    
    # 3. Collect Link Loads
    # Flatten the load matrix to get loads for each link
    # We only care about existing links.
    # The load matrix is symmetric? Let's check. 
    # Usually links are bidirectional in this model or handled as two directed edges. 
    # Graph is undirected in NetworkX but the routing code seems to handle indices.
    
    # Extract loads for valid edges
    loads = []
    link_names = []
    
    # Using the edges list defined at top
    for u, v in edges:
        u_idx = city_list.index(u)
        v_idx = city_list.index(v)
        # Load is sum of both directions? Or just one?
        # In the routing code: load_matrix[x-1][y-1] += traffic
        # It seems directional.
        # But the problem asks for "loads in all the links".
        # Assuming full duplex or just summing both directions for "link load".
        # Let's keep them separate or sum them?
        # Valid links are in 'edges'.
        
        l1 = final_load_matrix[u_idx][v_idx]
        l2 = final_load_matrix[v_idx][u_idx]
        
        # Storing as tuple or sum? Let's store individual directional loads or sum?
        # "Compute the loads ... in all the links". usually implies capacity usage.
        # Let's store both directions as separate bars or sum if it represents a single physical link capacity.
        # Given "Uncapacitated", usually we look at total flow.
        # Let's sum them for the "Link" load.
        total_link_load = l1 + l2
        loads.append(total_link_load)
        link_names.append(f"{u}-{v}")
        
    link_loads_data[strategy] = loads
    
    # Calculate balance metric (Standard Deviation)
    load_std = np.std(loads)
    results[strategy] = load_std
    print(f"Load Standard Deviation: {load_std:.2f}")

# ---------------------------------------------------------
# VISUALIZATION
# ---------------------------------------------------------
x = np.arange(len(edges))  # the label locations
width = 0.25  # the width of the bars

fig, ax = plt.subplots(figsize=(14, 7))
rects1 = ax.bar(x - width, link_loads_data['shortest'], width, label='Shortest')
rects2 = ax.bar(x, link_loads_data['longest'], width, label='Longest')
rects3 = ax.bar(x + width, link_loads_data['largest'], width, label='Largest')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Load (Gb/s)')
ax.set_title('Link Loads by Sorting Strategy')
ax.set_xticks(x)
ax.set_xticklabels(link_names, rotation=90)
ax.legend()

fig.tight_layout()
plt.show()

# ---------------------------------------------------------
# CONCLUSION
# ---------------------------------------------------------
print("\n=== CONCLUSION ===")
best_strategy = min(results, key=results.get)
print("Standard Deviation of Link Loads for each strategy:")
for s, v in results.items():
    print(f"  {s}: {v:.2f}")

print(f"\nThe best sorting strategy is '{best_strategy}' because it has the lowest standard deviation,")
print("indicating the most balanced distribution of traffic across the network links.")
