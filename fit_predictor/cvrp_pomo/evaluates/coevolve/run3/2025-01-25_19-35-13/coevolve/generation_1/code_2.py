import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the maximum distance in the distance matrix to use as a threshold
    max_distance = torch.max(distance_matrix)
    
    # Compute the normalized demand to get the demand per unit distance
    normalized_demand = demands / (max_distance + 1e-6)
    
    # Compute the heuristic value for each edge
    heuristics = normalized_demand * distance_matrix
    
    # Ensure that the heuristic values are within a reasonable range
    heuristics = torch.clamp(heuristics, min=-max_distance, max=max_distance)
    
    # Return the heuristic matrix
    return heuristics