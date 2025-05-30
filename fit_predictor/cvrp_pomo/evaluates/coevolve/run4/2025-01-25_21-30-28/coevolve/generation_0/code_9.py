import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total demand of all nodes
    total_demand = demands.sum()
    
    # Normalize the demand vector by the total vehicle capacity
    normalized_demands = demands / total_demand
    
    # Calculate the potential utility of each edge as the negative of the distance
    # multiplied by the normalized demand of the destination node
    potential_utility = -distance_matrix * normalized_demands
    
    # Optionally, you can add other heuristics such as:
    # - Add a small positive value for direct edges to the depot (distance to depot is 0)
    # - Subtract a larger value for edges with high demand (normalized demand > 1)
    # - Add a small positive value for edges with low demand (normalized demand < 1)
    
    # The heuristics matrix will have the same shape as the distance matrix
    heuristics_matrix = potential_utility
    
    return heuristics_matrix