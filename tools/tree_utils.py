# Contents of package:
# Classes and functions related to tree topologies utilities.
#
# (c) Paul Didier, SOUNDS ETN, KU Leuven ESAT STADIUS

import numpy as np
import networkx as nx
from collections import deque

    
def get_upstream_nodes(G: nx.Graph, root):
    """GPT"""
    # Compute shortest path distances from root
    dist = nx.single_source_shortest_path_length(G, root)
    
    upstreamNodes = [None for _ in range(len(G.nodes))]
    upstreamNeighbors = [None for _ in range(len(G.nodes))]
    for q in G.nodes:
        # Nodes "away" from root from the perspective of q:
        nodes_away = []

        # BFS from q, but only go to nodes with higher distance from root
        visited = set()
        stack = [q]
        
        while stack:
            node = stack.pop()
            visited.add(node)
            
            for neighbor in G.neighbors(node):
                if neighbor in visited:
                    continue
                if dist.get(neighbor, float('inf')) > dist[q]:
                    stack.append(neighbor)
                    nodes_away.append(neighbor)

        # Filter out q and root explicitly, if needed
        upstreamNodes[q] = np.sort([n for n in nodes_away if n != q and n != root])
        # Get the upstream neighbors of q
        upstreamNeighbors[q] = np.sort([n for n in G.neighbors(q) if n in upstreamNodes[q]])
    
    return upstreamNodes, upstreamNeighbors


def get_downstream_nodes(G: nx.Graph, root):
    nNodes = G.number_of_nodes()
    downstreamNeighbors = [None] * nNodes  # Initialize all entries to None
    downstreamNodes = [[] for _ in range(nNodes)]

    visited = set()
    queue = deque([root])
    visited.add(root)

    while queue:
        current = queue.popleft()
        for neighbor in G.neighbors(current):
            if neighbor not in visited:
                visited.add(neighbor)
                downstreamNeighbors[neighbor] = current  # current is "toward root" for neighbor
                queue.append(neighbor)
    
    # Find downstream nodes for each node
    for k in G.nodes:
        currIdx = k
        while downstreamNeighbors[currIdx] is not None:
            downstreamNodes[k].append(downstreamNeighbors[currIdx])
            currIdx = downstreamNeighbors[currIdx]

    return downstreamNodes, downstreamNeighbors


def generate_tree_with_diameter(N, E):
    """
    Generates a tree with N nodes and exact diameter E.
    
    Parameters:
        N (int): Number of nodes (N > E).
        E (int): Desired diameter (E >= 1).
    
    Returns:
        T (networkx.Graph): Tree with specified diameter.
    """
    assert E >= 1, "Diameter must be at least 1."
    assert N >= E + 1, "Need at least E+1 nodes for diameter E."

    tree = nx.Graph()

    # Step 1: create a path of length E (E+1 nodes)
    path_nodes = list(range(E + 1))
    for i in range(E):
        tree.add_edge(path_nodes[i], path_nodes[i+1])

    next_node = E + 1

    # Step 2: attach remaining nodes to internal nodes on the path
    attachable_nodes = path_nodes[1:-1]  # avoid attaching to endpoints to preserve diameter

    i = 0
    while next_node < N:
        attach_to = attachable_nodes[i % len(attachable_nodes)]
        tree.add_edge(attach_to, next_node)
        next_node += 1
        i += 1

    return tree


def tree_levels(G: nx.Graph, root):
    # Compute shortest path distances from root
    dist = nx.single_source_shortest_path_length(G, root)
    
    # Get the tree structure (list of leaves to root, per level)
    tree_structure = [[] for _ in range(max(dist.values()) + 1)]
    for q in G.nodes:
        tree_structure[dist[q]].append(q)
    tree_structure.reverse()  # Reverse the order to have root at the end
    
    return tree_structure