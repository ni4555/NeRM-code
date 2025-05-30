import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total demand
    total_demand = demands.sum()
    
    # Normalize the demands by the total vehicle capacity
    normalized_demands = demands / total_demand
    
    # Calculate the heuristic values for each edge
    # We use the negative of the normalized demand as the heuristic value
    # because we want to minimize the heuristic value to find the best edges
    heuristics = -normalized_demands * distance_matrix
    
    return heuristics