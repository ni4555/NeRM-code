import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the difference in demand between each pair of nodes
    demand_diff = demands.unsqueeze(1) - demands.unsqueeze(0)
    
    # Compute a simple heuristic based on the difference in demands
    # The heuristic promotes including edges where the difference in demand is close to zero
    # or positive, which indicates potential balance in load distribution
    heuristic_values = -torch.abs(demand_diff)
    
    # Incorporate distance matrix into the heuristic values to encourage shorter routes
    heuristic_values += distance_matrix
    
    # Normalize the heuristic values to ensure they are within a reasonable range
    # and to give a preference to edges with lower distances
    max_value = torch.max(heuristic_values)
    min_value = torch.min(heuristic_values)
    normalized_values = (heuristic_values - min_value) / (max_value - min_value)
    
    return normalized_values