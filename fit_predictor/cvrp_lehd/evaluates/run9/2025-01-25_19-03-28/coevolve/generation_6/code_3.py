import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the sum of demands to normalize them
    total_demand = demands.sum()
    
    # Normalize demands to the range [0, 1]
    normalized_demands = demands / total_demand
    
    # Calculate the load factor for each edge (i, j) as the sum of normalized demands
    # at customer nodes j that are reachable from i, excluding the depot (i.e., j != 0)
    load_factor = torch.zeros_like(distance_matrix)
    for i in range(distance_matrix.shape[0]):
        for j in range(1, distance_matrix.shape[1]):  # Skip the depot node
            load_factor[i, j] = (distance_matrix[i, j] < float('inf')).float() * normalized_demands[j]
    
    # The heuristic is a combination of the inverse distance and the load factor
    # Negative values are assigned to edges with high load factor to discourage their inclusion
    # The heuristic is positive for promising edges (low load factor, short distance)
    heuristic_matrix = distance_matrix.clone()
    heuristic_matrix[distance_matrix == float('inf')] = 0  # Set infinite distances to zero
    heuristic_matrix = 1 / heuristic_matrix + load_factor
    
    return heuristic_matrix