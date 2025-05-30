import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total demand
    total_demand = demands.sum()
    
    # Normalize the demands to be within the vehicle capacity range
    normalized_demands = demands / total_demand
    
    # Calculate the heuristics using the normalized demands
    # We can use a simple heuristic that promotes edges with lower normalized demand
    # and lower distance, as these are more likely to be part of the optimal route.
    # This is a simple heuristic and might not be the best for all cases, but it serves as an example.
    heuristics = -normalized_demands * distance_matrix
    
    # Clip the heuristics to ensure they are within the desired range
    heuristics = torch.clamp(heuristics, min=-1.0, max=1.0)
    
    return heuristics