import numpy as np
import numpy as np
import heapq

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    num_nodes = distance_matrix.shape[0]
    edge_list = []

    # Create edge list of tuples (weight, start, end)
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            weight = distance_matrix[i][j]
            edge_list.append((weight, i, j))

    # Kruskal's algorithm to find the MST
    def find(parent, i):
        if parent[i] == i:
            return i
        return find(parent, parent[i])

    def union(parent, rank, x, y):
        xroot = find(parent, x)
        yroot = find(parent, y)
        if rank[xroot] < rank[yroot]:
            parent[xroot] = yroot
        elif rank[xroot] > rank[yroot]:
            parent[yroot] = xroot
        else:
            parent[yroot] = xroot
            rank[xroot] += 1

    parent = []
    rank = []

    for node in range(num_nodes):
        parent.append(node)
        rank.append(0)

    # Sort the edges in ascending order of their weight
    edge_list.sort()

    mst_edges = []
    for weight, u, v in edge_list:
        if find(parent, u) != find(parent, v):
            union(parent, rank, u, v)
            mst_edges.append((u, v))

    # Calculate MST weight
    mst_weight = sum(distance_matrix[u][v] for u, v in mst_edges)

    # Create the heuristic matrix
    heuristic_matrix = np.zeros_like(distance_matrix)
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                # The penalty is the edge weight minus the MST weight
                penalty = distance_matrix[i][j] - mst_weight
                heuristic_matrix[i][j] = max(penalty, 0)

    return heuristic_matrix

# Example usage:
# distance_matrix = np.array([[0, 1, 3, 2],
#                             [1, 0, 2, 3],
#                             [3, 2, 0, 1],
#                             [2, 3, 1, 0]])
# print(heuristics_v2(distance_matrix))