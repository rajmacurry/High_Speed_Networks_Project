# -*- coding: utf-8 -*-
"""
@authors: Jose Ribeiro (Graph class, getPaths, shortestPaths, countHops and create_traffic_matrix)
          Alexandre Freitas (orderPaths, create_load_matrix, update_network, breakTie, route, route_path, hop_count and printResults)
"""

import numpy as np
import copy
import itertools
import time

# Class to represent a graph
class Graph:

    # A utility function to find the
    # vertex with minimum dist value, from
    # the set of vertices still in queue
    def minDistance(self, dist, queue):
        # Initialize min value and min_index as -1
        minimum = float("Inf")
        min_index = -1

        # from the dist array,pick one which
        # has min value and is till in queue
        for i in range(len(dist)):
            if dist[i] < minimum and i in queue:
                minimum = dist[i]
                min_index = i
        return min_index

    # Function to print shortest path
    # from source to j
    # using parent array
    def printPath(self, parent, j):
        path = []
        # Base Case : If j is source
        if parent[j] == -1:
            # print (j+1)
            return [j + 1]

        path.extend(self.printPath(parent, parent[j]))
        # print (j+1)
        path.append(j + 1)
        return path

        # A utility function to print

    # the constructed distance
    # array
    def printSolution(self, src, dist, parent):
        paths = []
        # print("Vertex \t\tDistance from Source\tPath")
        for i in range(0, len(dist)):
            # print("\n%d --> %d \t\t%d \t\t\t\t\t" % (src+1, i+1, dist[i])),
            path = self.printPath(parent, i)
            paths.append({
                "source": src + 1,
                "destination": i + 1,
                "distance": dist[i],
                "path": path
            })
        return paths

    '''Function that implements Dijkstra's single source shortest path
    algorithm for a graph represented using adjacency matrix
    representation'''

    def dijkstra(self, graph, src):

        row = len(graph)
        col = len(graph[0])

        # The output array. dist[i] will hold
        # the shortest distance from src to i
        # Initialize all distances as INFINITE
        dist = [float("Inf")] * row

        # Parent array to store
        # shortest path tree
        parent = [-1] * row

        # Distance of source vertex
        # from itself is always 0
        dist[src] = 0

        # Add all vertices in queue
        queue = []
        for i in range(row):
            queue.append(i)

        # Find shortest path for all vertices
        while queue:

            # Pick the minimum dist vertex
            # from the set of vertices
            # still in queue
            u = self.minDistance(dist, queue)

            # remove min element
            if u != -1:
                queue.remove(u)

                # Update dist value and parent
                # index of the adjacent vertices of
                # the picked vertex. Consider only
                # those vertices which are still in
                # queue
                for i in range(col):
                    '''Update dist[i] only if it is in queue, there is
                    an edge from u to i, and total weight of path from
                    src to i through u is smaller than current value of
                    dist[i]'''
                    if graph[u][i] and i in queue:
                        if dist[u] + graph[u][i] < dist[i]:
                            dist[i] = dist[u] + graph[u][i]
                            parent[i] = u
            else:
                queue.clear()

        # print the constructed distance array
        return self.printSolution(src, dist, parent)


def getPaths(graph: Graph, matrix: list):
    patths = []
    # Print the solution
    for i in range(len(matrix)):
        patths.append(graph.dijkstra(matrix, i))

    return patths


def shortestPaths(graph: Graph, matrix: list):
    pairs = []
    paths = []

    #count = min([len([i for i in row if i > 0]) for row in matrix])
    count = 2
    for i in range(0, len(matrix)):
        for j in range(i, len(matrix)):
            if matrix[i][j] != 0:
                pairs.append(f'{i + 1}-{j + 1}')

    for i in range(len(matrix)):
        if i > count:
            break
        combinationss = list(itertools.combinations(pairs, i))

        for comb in combinationss:
            aux_matrix = copy.deepcopy(matrix)

            for pair in comb:
                row, column = pair.split('-')
                aux_matrix[int(row) - 1][int(column) - 1] = 0
                aux_matrix[int(column) - 1][int(row) - 1] = 0

            # print(f'Pair removed: {comb}')
            # for row in range(len(np.array(aux_matrix))):
            # print(aux_matrix[row])

            if not (~np.array(aux_matrix).any(axis=0)).any():
                aux_paths = getPaths(graph, aux_matrix)
                if len(paths) == 0:
                    paths = aux_paths
                    for path in paths:
                        for p in path:
                            p["path"] = [p["path"]]
                else:
                    for a, path in enumerate(paths):
                        for b, p in enumerate(path):
                            if aux_paths[a][b]["distance"] == p["distance"] and aux_paths[a][b]["path"] not in p[
                                    "path"]:
                                p["path"].append(aux_paths[a][b]["path"])
    return paths

def countHops(paths: list):
    hops_matrix = []

    for path in paths:
        hops_row = []
        for p in path:
            hops_row.append(len(min(p["path"], key=len)) - 1)
        hops_matrix.append(hops_row)

    return hops_matrix

def create_traffic_matrix(matrix, traffic):
    matrix_size = len(matrix)
    if traffic == None or len(traffic) != matrix_size:
        a = np.ones((matrix_size, matrix_size), int)
        np.fill_diagonal(a, 0)
        return a.tolist()
    else:
        return traffic

# Returns the ordered traffic demands according to a given sorting strategy 
# (shortest, longest or largest) based on the distance.     
def orderPaths(paths: list, traffic_matrix: list, hop_matrix: list, order = "shortest"):
    
    path_list = []
    # Iterate over the shortest-paths list.
    for path in paths:
        for p in path:
            p["traffic"] = traffic_matrix[p["source"]-1][p["destination"]-1]
            p["hops"] = hop_matrix[p["source"]-1][p["destination"]-1]
            p['routed'] = False
            
            # If there are multiple shortest-paths, remove the ones with larger
            # number of hops.
            possible_paths = p["path"]
            if len(possible_paths) > 1:
                length_min_path = min(len(x) for x in possible_paths)
                possible_paths_aux = possible_paths.copy()
                for idx, item in enumerate(possible_paths):
                    if len(item) > length_min_path:
                        possible_paths_aux.remove(item)
                        p['path'] = possible_paths_aux
            
            # Include only traffic demands with source different than destination
            # and with traffic.
            if p["source"] != p["destination"] and p["traffic"] > 0:
                path_list.append(p)
    
    # Order the paths.    
    if order == "shortest":
        ordered_paths = sorted( path_list, key = lambda d: (d['distance'], len(d['path'])) )
    elif order == "longest":
        ordered_paths = sorted( path_list, key = lambda d: (d['distance'], -len(d['path'])),reverse = True )
    elif order == "largest":
        ordered_paths = sorted( path_list, key = lambda d: (d['traffic'], -len(d['path'])),reverse = True )
    else:
        ordered_paths = sorted( path_list, key = lambda d: (d['distance'], len(d['path'])) )

    return ordered_paths

# Create NxN matrix to represent the load of each one of the links in the network
def create_load_matrix(matrix):
    matrix_size = len(matrix)
    a = np.zeros((matrix_size, matrix_size), int)
    return a.tolist()

# Updates the network by removing links that have become saturated with traffic and then
# recalculates the shortest-paths. Called on route function.
def update_network(links_to_remove,matrix):
    # Iterate over each link to be removed and set the length 
    # of the link in the network to a large value (virtually removes it).
    for link in links_to_remove:
        matrix[link[0]-1][link[1]-1] = 99999
    
    # Recalculate the shortest paths.
    new_shortest_paths = shortestPaths(graph,matrix)

    # If there are multiple shortest paths, remove the ones with larger
    # number of hops.
    for path in new_shortest_paths:
        for p in path:            
            possible_paths = p["path"]
            if len(possible_paths) > 1:
                length_min_path = min(len(x) for x in possible_paths)
                possible_paths_aux = possible_paths.copy()
                for idx, item in enumerate(possible_paths):
                    if len(item) > length_min_path:
                        possible_paths_aux.remove(item)
                        p['path'] = possible_paths_aux
                
    return new_shortest_paths

# Function that, if there are multiple paths to the destination, returns the one 
# that minimizes the maximum load between the links.
def breakTie(p:dict,load_matrix:list):
    loads_of_path = []
    all_loads = [None] * len (p['path'])
    # Determines loads for each path.
    for idx,_ in enumerate(p['path']):
        for x,y in zip(p['path'][idx],p['path'][idx][1:]):
            loads_of_path.append(load_matrix[x-1][y-1])
            
        all_loads[idx]= copy.deepcopy(loads_of_path)
        loads_of_path.clear()
        # all_loads is a list of lists where each inner list has the loads in each link of a path.

    maximum_list = []
    # After having all the loads in all the paths, find the path with the minimum maximum load.
    while True:
        for path_load in all_loads:
            # If all loads in all paths are the same in the end, just pick the first path in the list (index 0).
            if len(path_load) == 0:
                chosen_path = 0
                return chosen_path  
            
            maximum_list.append(max(path_load))
            path_load.remove(max(path_load))
        
        # If all elements in maximum_list are the same (there's no minimum), 
        # move on to the links with next largest loads.
        if all(ele == maximum_list[0] for ele in maximum_list):
            maximum_list.clear()
            continue
        # Not all equal, there's a minimum:
        else:
            # Check if min is unique. If so, the path was found, else go to next largest loads.
            min_val = min(maximum_list)
            if maximum_list.count(min_val) == 1:
                chosen_path = maximum_list.index(min_val)
                return chosen_path
            else:
                maximum_list.clear()
                continue

# Auxiliary function to the route function, updates the matrices (load_matrix, path_matrix) 
# and checks if there are links that need to be removed (adds them to a list) and sets
# update_net_flag to True in case a link has residual capacity zero.
def route_path(p, load_matrix, path_matrix, chosen_path, p_aux, links_to_remove):
    update_net_flag = False
    blocked_flag = False
    
    # Case where traffic has value greater than one.
    if p_aux['traffic'] > 1:
        # Check all links to see if traffic fits the entire path, if not set blocked_flag as True.
        for x,y in zip(p['path'][chosen_path],p['path'][chosen_path][1:]):
            if load_matrix[x-1][y-1] + p_aux['traffic'] > MAX_LINK_CAP:
                blocked_flag = True
        
        # Case the traffic does not fit the path (is blocked):
        if blocked_flag:
            return update_net_flag, blocked_flag
    
    # Update the matrices accordingly.
    path_matrix[p['source']-1][p['destination']-1] = p['path'][chosen_path]
    # Iterate over all edges of the chosen path:
    for x,y in zip(p['path'][chosen_path],p['path'][chosen_path][1:]):
        load_matrix[x-1][y-1] += p_aux['traffic']
        # Case of link saturation
        if load_matrix[x-1][y-1] == MAX_LINK_CAP:
            update_net_flag = True
            links_to_remove.append((x,y))

    return update_net_flag, blocked_flag

# Routes the traffic according to ordered_traffic_demands. Returns the completed load_matrix, path_matrix,
# distance_matrix, blocked_traffic (number of blocked paths), and blocked_paths (list of blocked paths).
def route(ordered_traffic_demands, load_matrix:list,matrix):
    # Initialize matrices and lists
    a = np.zeros((len(load_matrix),len(load_matrix)),int)
    path_matrix = a.tolist()
    distance_matrix = a.tolist()
    
    new_paths_flag = False
    blocked_traffic = 0
    blocked_paths = []
    links_to_remove = []
    
    # Iterate on the ordered list of traffic demands.
    for p in ordered_traffic_demands:
        if p['routed'] == True:
            continue
        
        p['routed'] = True
        
        links_to_remove = []
        
        # p_aux is the p in the original ordered_traffic_demands, 
        # because the recalculated paths do not have 'traffic' in dictionary.
        p_aux = p
        
        # In case there was a recalculation of shortest-paths in every iteration p
        # is now from the new_shortest_paths list.
        if new_paths_flag:
            p = new_shortest_paths[p['source']-1][p['destination']-1]

        distance_matrix[p['source']-1][p['destination']-1] = p['distance']
        
        # In case a path does not exist (path_matrix value will be kept at 0)
        if p['distance'] >= 99999:
            blocked_traffic += p_aux['traffic']
            blocked_paths.append(p)
            continue
        
        # Choose the path index of the path to route through.
        # Only a single path between source and destination.
        if len(p['path']) == 1:
            chosen_path = 0
        # In case there is a path tie (two or more paths between source and destination)
        else:
            chosen_path = breakTie(p,load_matrix)

        update_net_flag, blocked_flag = route_path(p, load_matrix, path_matrix, chosen_path, p_aux,links_to_remove)
        # In case traffic > 1, blocked traffic can occur inside each iteration.
        if blocked_flag:
            distance_matrix[p['source']-1][p['destination']-1] = 0
            blocked_traffic += p_aux['traffic']
            blocked_paths.append(p)
            continue
        
        # So that symmetric connections are made one after the other and through the same path.
        for p2 in ordered_traffic_demands:
            if p2['source'] == p['destination'] and p2['destination'] == p['source'] and p_aux['traffic'] == p2['traffic'] and p2['routed'] == False:
                
                p2['routed'] = True

                if new_paths_flag:
                    p2 = new_shortest_paths[p2['source']-1][p2['destination']-1]

                distance_matrix[p2['source']-1][p2['destination']-1] = p2['distance']
                
                # Choose same path:
                chosen_path2 = 0
                while True:    
                    # check if path of p contains all elements of current path of p2
                    result = all(elem in p['path'][chosen_path] for elem in p2['path'][chosen_path2])
                    if result == True:
                        break
                    else:
                        chosen_path2 += 1

                update_net_flag, _ = route_path(p2, load_matrix, path_matrix, chosen_path2, p_aux, links_to_remove)
                
                break
                
        # If there is a link that has reached the limit capacity, network must be updated 
        # (remove the links that are saturated from matrix and find new shortest-paths).
        if update_net_flag:
            new_shortest_paths = update_network(links_to_remove,matrix)
            new_paths_flag = True
            
    return load_matrix, path_matrix, distance_matrix, blocked_traffic, blocked_paths

# Prints the ordered traffic demands and loads in every link and calculates average load per link.
def printResults (ordered_paths:list, adj_matrix:list,load_matrix: list,distance_matrix:list):
    print("Ordered Paths:")
    for p in ordered_paths:
        if p['routed'] == True:
            if path_matrix[p['source']-1][p['destination']-1] != 0:
                print("Path: {} , Dist: {} , Og.Dist: {} , traffic = {} (Og. Path: {})".format(path_matrix[p['source']-1][p['destination']-1],
                                                                                                       distance_matrix[p['source']-1][p['destination']-1],
                                                                                                       p['distance'], traffic[p['source']-1][p['destination']-1],p['path']))
            elif path_matrix[p['source']-1][p['destination']-1] == 0:
                print("Path: BLOCKED , Dist: BLOCKED , Og.Dist: {} , traffic = {} (Og. Path: {})".format(p['distance'],traffic[p['source']-1][p['destination']-1],p['path']))
            p['routed'] = False
        for p2 in ordered_paths:
            if p2['source'] == p['destination'] and p2['destination'] == p['source'] and p['traffic'] == p2['traffic'] and p2['routed'] == True:
                if path_matrix[p2['source']-1][p2['destination']-1] != 0:
                    print("Path: {} , Dist: {} , Og.Dist: {} , traffic = {} (Og. Path: {})".format(path_matrix[p2['source']-1][p2['destination']-1],
                                                                                                     distance_matrix[p2['source']-1][p2['destination']-1], p2['distance'],
                                                                                                     traffic[p2['source']-1][p2['destination']-1],p2['path']))
                elif path_matrix[p2['source']-1][p2['destination']-1] == 0:
                    print("Path: BLOCKED , Dist: BLOCKED , Og.Dist: {} , traffic = {} (Og. Path: {})".format(p2['distance'],traffic[p2['source']-1][p2['destination']-1],p2['path']))
                #print("Path: {} -> traffic = {}".format(path_matrix[p2['source']-1][p2['destination']-1],traffic[p2['source']-1][p2['destination']-1]))
                p2['routed'] = False 
   
    accum_sum = 0
    n_links = 0
    print("---LOADS IN EVERY LINK---")
    for row,_ in enumerate(load_matrix):
        for collumn,_ in enumerate (load_matrix[row]):
            if adj_matrix[row][collumn] == 0:
                continue
            print("{}-{} -> {}".format(row+1,collumn+1,load_matrix[row][collumn]))
            #print("{}->{}".format(row+1,collumn+1))
            #print(load_matrix[row][collumn])
            accum_sum += load_matrix[row][collumn]
            n_links += 1
    
    average_load_per_link = accum_sum/n_links
    return average_load_per_link

# Returns hop_matrix and avg_hops_per_path calculated through the path_matrix, 
# (the returned values refer to the final chosen paths after routing).
def hop_count(path_matrix):
    hop_matrix = np.zeros((len(path_matrix),len(path_matrix)))
    for row,_ in enumerate(path_matrix):
        for collumn,_ in enumerate (path_matrix[row]):
            if path_matrix[row][collumn] == 0:
                continue

            hop_matrix[row][collumn] = len(path_matrix[row][collumn]) - 1
    
    avg_hops_per_path = np.mean(hop_matrix[np.nonzero(hop_matrix)])
    
    return hop_matrix,avg_hops_per_path

# Function to check if matrix is symmetric 
# (with a tolerance due to the limitations of floating point precision)  
def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)

# Function to check if all elements on main diagonal are zero. Return True if so.
def check_diagonal_zero(matrix):
    diagonal = np.diagonal(matrix)
    return np.all(diagonal == 0)

graph = Graph()
average_node_degree = []
traffic = []
number_of_hops_per_demand = []
diameter = []

# Default value for the link capacity (999999 corresponds to uncapacitated routing)
MAX_LINK_CAP = 999999
#MAX_LINK_CAP = 64
#MAX_LINK_CAP = 75
#------------------------------------INPUT MATRICES HERE------------------------------------------------
# If traffic matrix is not defined, the routing will be done using a full-mesh logical topology with
# one unit of traffic.
#TEST NETWORK
MAX_LINK_CAP = 5
matrix = [[0,10,10,30,0,0,0,0],
          [10,0,10,0,0,30,20,0],
          [10,10,0,10,10,0,40,0],
          [30,0,10,0,0,0,0,0],
          [0,0,10,0,0,0,0,20],
          [0,30,0,0,0,0,10,0],
          [0,20,40,0,0,10,0,10],
          [0,0,0,0,20,0,10,0]]
'''traffic = [[0, 10, 10, 10, 10, 10, 10, 10], 
           [10, 0, 1, 1, 1, 1, 1, 1], 
           [10, 1, 0, 1, 1, 1, 1, 1], 
           [10, 1, 1, 0, 1, 1, 1, 1], 
           [10, 1, 1, 1, 0, 1, 1, 1], 
           [10, 1, 1, 1, 1, 0, 1, 1], 
           [10, 1, 1, 1, 1, 1, 0, 1], 
           [10, 1, 1, 1, 1, 1, 1, 0]]
'''


#CESNET
'''matrix = [[0,226.07,334.4,0,0,0,274.08],
          [226.07,0,315.98,0,0,0,0],
          [334.4,315.98,0,425.25,0,0,0],
          [0,0,425.25,0,378.51,173.75,0],
          [0,0,0,378.51,0,212.79,0],
          [0,0,0,173.75,212.79,0,330.72],
          [274.08,0,0,0,0,330.72,0]]'''
'''traffic = [[0,119.63,125.95,94.72,59.12,91.55,163.3],
          [119.63,0,95.83,83.72,54.12,79.97,114.80],
          [125.95,95.83,0.00,80.46,81.02,0.00,104.33],
          [94.72,83.72,80.46,0.00,116.99,144.81,114.30],
          [59.12,54.12,81.02,116.99,0.00,124.85,110.11],
          [91.55,79.97,0.00,144.81,124.85,0.00,144.93],
          [163.30,114.80,104.33,114.30,110.11,144.93,0.00]]
'''


#------------------------ INPUT SORTING ORDER----------------------------------------------
sorting_order = "shortest"
#sorting_order = "longest"
#sorting_order = "largest"

#-------------------------DETERMINATION OF NETWORK PARAMETERS-----------------------------

matrix_np = np.array(matrix,dtype=float)
#print(np.shape(matrix_np))
if not check_symmetric(matrix_np):
    print("ERROR: The input adjacency matrix (matrix) is not symmetric.")
    exit()
if not check_diagonal_zero(matrix_np):
    print("ERROR: The input adjacency matrix (matrix) has an element on the main diagonal different than zero.")
    exit()

start = time.time()

paths = shortestPaths(graph, matrix)

hop_matrix = countHops(paths)
average_node_degree.append(np.count_nonzero(matrix)/len(matrix))

N = matrix_np.shape[0]
number_of_links = np.count_nonzero(matrix_np)/2
min_link_length = np.min(matrix_np[np.nonzero(matrix_np)])
max_link_length = np.max(matrix_np)
matrix_np[matrix_np == 0] = np.nan
avg_link_length = np.nanmean(matrix_np)

traffic = create_traffic_matrix(matrix, traffic)
number_of_hops_per_demand.append(np.matrix(hop_matrix).sum() / np.matrix(traffic).sum())
diameter.append(np.matrix(hop_matrix).max())

ordered_paths = orderPaths(paths,traffic,hop_matrix, sorting_order)
load_matrix = create_load_matrix (matrix)
matrix_cp = matrix.copy()
load_matrix,path_matrix,distance_matrix,blocked_traffic, blocked_paths= route(ordered_paths, load_matrix, matrix_cp)

distance_matrix_np = np.array(distance_matrix,dtype=float)
distance_matrix_np[distance_matrix_np == 0] = np.nan
distance_matrix_np[distance_matrix_np >= 99999] = np.nan
min_path_length = np.nanmin(distance_matrix_np)
max_path_length = np.nanmax(distance_matrix_np)
avg_path_length = np.nanmean(distance_matrix_np)

total_num_paths = len(ordered_paths)
blocking_prob = len(blocked_paths)/total_num_paths

final_hop_matrix, avg_hops_per_path = hop_count(path_matrix)


# PRINTING RESULTS:
print("Number of Nodes:")
print(N)
print("Number of links:")
print(number_of_links)
print("Average Node Degree:")
print(average_node_degree)
print("Network Diameter:")
print(diameter)
print("Average Number of Hops per Demand:")
print(number_of_hops_per_demand)
print("Minimum link length:")
print(min_link_length)
print("Maximum link length:")
print(max_link_length)
print("Average link length:")
print(avg_link_length)
print("Total Number of Paths:")
print(len(ordered_paths))

average_load_per_link = printResults(ordered_paths,matrix,load_matrix,distance_matrix)

print("Average load per link:")
print(average_load_per_link)

print("Minimum path length:")
print(min_path_length)
print("Maximum path length:")
print(max_path_length)
print("Average path length:")
print(avg_path_length)
print("Average Number of Hops per Path (after routing):")
print(avg_hops_per_path)

print("Blocked Traffic:")
print(blocked_traffic)
print("Blocking Probability:")
print(blocking_prob)

for b in blocked_paths:
    print("'source':{},'destination':{},'path':{}".format(b['source'],b['destination'],b['path']))

end = time.time()
print("Runtime:")
print(end - start)