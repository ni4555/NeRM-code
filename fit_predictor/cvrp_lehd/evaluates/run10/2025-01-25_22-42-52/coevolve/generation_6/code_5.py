import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize demands by total vehicle capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity
    
    # Compute the sum of demands along the diagonal
    sum_of_demands = torch.diag(normalized_demands)
    
    # Compute the heuristic value for each edge
    # Heuristic value = demand at the destination node - sum of demands along the path
    heuristic_values = demands - sum_of_demands
    
    # Adjust heuristic values based on distance
    # Negative values for undesirable edges (e.g., long distances)
    # Positive values for promising edges (e.g., short distances)
    adjusted_heuristic_values = heuristic_values * distance_matrix
    
    return adjusted_heuristic_values