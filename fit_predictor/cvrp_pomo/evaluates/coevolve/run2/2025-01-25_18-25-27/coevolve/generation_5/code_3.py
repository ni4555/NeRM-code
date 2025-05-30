import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total vehicle capacity
    total_capacity = demands.sum()
    
    # Normalize the demands by the total vehicle capacity
    normalized_demands = demands / total_capacity
    
    # Calculate the heuristic values
    # The heuristic is a combination of the normalized demand and the distance
    # We use a negative value for the distance to ensure that edges with higher distances are less promising
    heuristics = -distance_matrix + normalized_demands.unsqueeze(1) * total_capacity
    
    return heuristics