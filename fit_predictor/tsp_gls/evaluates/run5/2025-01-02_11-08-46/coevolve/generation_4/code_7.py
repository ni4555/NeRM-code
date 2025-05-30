import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Initialize the heuristics array with zeros
    heuristics = np.zeros_like(distance_matrix, dtype=float)
    
    # Distance-weighted normalization
    max_distance = np.max(distance_matrix)
    min_distance = np.min(distance_matrix)
    heuristics = (distance_matrix - min_distance) / (max_distance - min_distance)
    
    # Resilient minimum spanning tree heuristic
    # We use Kruskal's algorithm for simplicity, which is efficient for dense graphs
    sorted_edges = np.argsort(distance_matrix)
    num_nodes = distance_matrix.shape[0]
    parent = list(range(num_nodes))
    rank = [0] * num_nodes
    
    def find_set(node):
        if parent[node] != node:
            parent[node] = find_set(parent[node])
        return parent[node]
    
    def union(node1, node2):
        root1 = find_set(node1)
        root2 = find_set(node2)
        if root1 != root2:
            if rank[root1] > rank[root2]:
                parent[root2] = root1
            elif rank[root1] < rank[root2]:
                parent[root1] = root2
            else:
                parent[root2] = root1
                rank[root1] += 1
    
    # Apply Kruskal's algorithm
    for edge in sorted_edges:
        node1, node2 = edge // num_nodes, edge % num_nodes
        if find_set(node1) != find_set(node2):
            union(node1, node2)
            heuristics[edge] = 1.0  # Mark the edge as included in the minimum spanning tree
    
    return heuristics