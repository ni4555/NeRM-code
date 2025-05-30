import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the cumulative demand for each edge
    # We use the fact that the demand vector is normalized by the total vehicle capacity
    # Thus, we can directly use it as weights for the edges
    edge_demands = demands.unsqueeze(1) * demands.unsqueeze(0)
    
    # Invert the cumulative demand to create a heuristic value
    # Negative values are undesirable, positive values are promising
    heuristics = -edge_demands
    
    # Add a small constant to avoid division by zero in the next step
    heuristics += 1e-8
    
    # Normalize the heuristics by the distance to ensure that shorter distances are more promising
    # This is a common approach in many heuristics
    heuristics /= distance_matrix
    
    return heuristics