import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the sum of demands to normalize them
    total_capacity = demands.sum()
    
    # Normalize the demands by the total vehicle capacity
    normalized_demands = demands / total_capacity
    
    # Calculate the potential negative impact of each edge (high demand or high distance)
    negative_impact = -torch.abs(normalized_demands) - distance_matrix
    
    # Calculate the potential positive impact of each edge (low demand and low distance)
    positive_impact = torch.min(normalized_demands, 1 - normalized_demands) - distance_matrix
    
    # Combine the negative and positive impacts to form the heuristic values
    heuristic_values = negative_impact + positive_impact
    
    # Ensure that the heuristic values are within the range [-1, 1] for better performance
    heuristic_values = torch.clamp(heuristic_values, min=-1, max=1)
    
    return heuristic_values