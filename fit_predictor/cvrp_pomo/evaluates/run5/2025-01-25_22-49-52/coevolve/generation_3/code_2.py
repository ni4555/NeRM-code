import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Get the size of the distance matrix
    n = distance_matrix.shape[0]
    
    # Normalize the demands by the sum of demands (total vehicle capacity)
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity
    
    # Initialize the heuristic matrix with zeros
    heuristic_matrix = torch.zeros_like(distance_matrix)
    
    # Node partitioning: group nodes based on their demand
    sorted_indices = torch.argsort(normalized_demands)
    threshold = 0.5  # threshold for partitioning nodes
    partitions = torch.zeros(n, dtype=torch.bool)
    for i in range(n):
        partitions[sorted_indices[i]] = normalized_demands[sorted_indices[i]] > threshold
    
    # Demand relaxation: adjust demands for partitioned nodes
    relaxed_demands = demands.clone()
    relaxed_demands[partitions] = 1.0
    
    # Path decomposition: calculate potential savings for each edge
    savings_matrix = distance_matrix.clone()
    savings_matrix[partitions[:, None] & partitions] -= (relaxed_demands[:, None] * relaxed_demands)
    
    # Dynamic window approach: update heuristics based on path decomposition
    for i in range(n):
        for j in range(n):
            if i != j:
                heuristic_matrix[i, j] = savings_matrix[i, j] - distance_matrix[i, j]
    
    # Apply the multi-objective evolutionary algorithm principle:
    # penalize edges that lead to unbalanced loads
    load_balance_penalty = torch.abs((relaxed_demands - 1.0).sum() / total_capacity)
    heuristic_matrix += load_balance_penalty
    
    return heuristic_matrix