import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the sum of the demands to normalize
    total_demand = demands.sum()
    
    # Normalize the demands by the total vehicle capacity
    normalized_demands = demands / total_demand
    
    # Calculate the heuristic values as negative of the distance matrix times the normalized demands
    # This heuristic assumes that shorter distances and lower demands are more promising
    heuristics = -distance_matrix * normalized_demands
    
    return heuristics