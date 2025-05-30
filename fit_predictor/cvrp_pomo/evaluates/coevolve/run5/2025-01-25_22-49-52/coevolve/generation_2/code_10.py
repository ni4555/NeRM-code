import random
import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    # Initialize a tensor with zeros
    heuristics = torch.zeros_like(distance_matrix)
    
    # Node partitioning
    # For simplicity, use a simple partitioning based on demand
    threshold = 0.5 * demands.sum() / n
    partitions = torch.zeros(n)
    for i in range(n):
        partitions[i] = 1 if demands[i] > threshold else 0
    
    # Demand relaxation
    relaxed_demands = demands * (1 - partitions)
    
    # Path decomposition
    # For simplicity, use a greedy approach to decompose paths
    for i in range(n):
        if relaxed_demands[i] > 0:
            # Find the nearest node to i with relaxed_demand > 0
            nearest_node = torch.argmin(distance_matrix[i, ~torch.isnan(demands)])
            # Update the heuristic for the edge (i, nearest_node)
            heuristics[i, nearest_node] += 1
    
    # Multi-objective evolutionary algorithm
    # For simplicity, use a random approach to adjust the heuristics
    for _ in range(10):  # Number of iterations
        # Randomly select a fraction of edges and adjust their heuristic values
        num_edges_to_adjust = int(0.1 * n * n)
        indices_to_adjust = torch.randperm(n * n)[:num_edges_to_adjust]
        for index in indices_to_adjust:
            i, j = divmod(index, n)
            heuristics[i, j] += torch.randn(1)
    
    # Dynamic window technique
    # For simplicity, adjust the heuristics based on the distance matrix
    for i in range(n):
        for j in range(n):
            if i != j:
                # Increase the heuristic value if the distance is short
                heuristics[i, j] += (1 - distance_matrix[i, j] / distance_matrix.max())
    
    return heuristics