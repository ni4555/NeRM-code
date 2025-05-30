import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the sum of demands to normalize them
    total_demand = torch.sum(demands)
    
    # Normalize the demands
    normalized_demands = demands / total_demand
    
    # Create a matrix to hold the heuristic values
    heuristics_matrix = torch.full_like(distance_matrix, fill_value=-1e6)
    
    # Calculate the heuristics for each edge
    # Subtracting the normalized demand of the destination from the normalized demand of the source
    for i in range(distance_matrix.shape[0]):
        for j in range(distance_matrix.shape[1]):
            if i != j:  # Avoid self-loops
                heuristics_matrix[i, j] = normalized_demands[i] - normalized_demands[j]
    
    return heuristics_matrix