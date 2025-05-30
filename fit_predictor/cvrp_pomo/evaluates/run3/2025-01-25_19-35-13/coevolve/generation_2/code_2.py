import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the sum of demands normalized by the total vehicle capacity
    total_demand = demands.sum()
    demand_normalized = demands / total_demand
    
    # Create a matrix to store heuristics values
    heuristics_matrix = torch.zeros_like(distance_matrix)
    
    # Populate the heuristics matrix
    # For each edge (i, j), calculate the heuristics as:
    # 1 - (distance from i to j / max possible distance) * (1 - demand_normalized[j])
    for i in range(distance_matrix.size(0)):
        for j in range(distance_matrix.size(1)):
            if i != j:  # Avoid the depot itself
                heuristics_matrix[i, j] = 1 - (distance_matrix[i, j] / distance_matrix.max()) * (1 - demand_normalized[j])
    
    return heuristics_matrix