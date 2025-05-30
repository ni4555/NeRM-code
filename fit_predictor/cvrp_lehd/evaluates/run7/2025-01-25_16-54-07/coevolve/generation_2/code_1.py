import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the normalized demands
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity
    
    # Calculate the potential of each edge based on demand and distance
    # We use a simple heuristic where the potential is inversely proportional to the distance
    # and also takes into account the normalized demand of the destination node.
    # Negative values are assigned to edges with higher distance or lower demand.
    potential = -distance_matrix + normalized_demands.unsqueeze(1) * distance_matrix
    
    return potential