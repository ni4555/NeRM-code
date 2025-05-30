import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Ensure that the demands are normalized by the total vehicle capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity

    # Initialize the potential value matrix with a high value
    num_nodes = distance_matrix.shape[0]
    potential_matrix = torch.full((num_nodes, num_nodes), fill_value=1e9)

    # Define the potential function for each edge
    def potential(a, b):
        return distance_matrix[a, b] - demands[a] - demands[b]

    # Node partitioning to divide the nodes into clusters
    # Placeholder for actual node partitioning logic
    clusters = torch.zeros(num_nodes, dtype=torch.long)
    for i in range(num_nodes):
        clusters[i] = i % 2  # Example partitioning: even-indexed nodes in one cluster

    # Demand relaxation within clusters
    for cluster in torch.unique(clusters):
        cluster_nodes = clusters == cluster
        demands_in_cluster = normalized_demands[cluster_nodes]
        potential_matrix[cluster_nodes, cluster_nodes] -= demands_in_cluster.sum()

    # Path decomposition to calculate potential values for each node
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                potential_matrix[i, j] = min(potential_matrix[i, j], potential(i, j))

    # Normalize the potential matrix to get the heuristic values
    min_potential = potential_matrix.min()
    max_potential = potential_matrix.max()
    normalized_potential_matrix = (potential_matrix - min_potential) / (max_potential - min_potential)

    # Subtract the normalized potential matrix from 1 to get heuristic values
    # Negative values for undesirable edges, positive values for promising ones
    heuristics = 1 - normalized_potential_matrix

    return heuristics