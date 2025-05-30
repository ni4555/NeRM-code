import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize demands by the total vehicle capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity
    
    # Compute the negative of the demand to make high demand edges less promising
    negative_demand = -normalized_demands
    
    # Create a heuristics matrix by combining the negative demand with the distance matrix
    heuristics_matrix = negative_demand.unsqueeze(1) + distance_matrix
    
    # Use element-wise max to give more weight to high demand edges
    # The max will also ensure that there are no negative values, as we are using max of two
    # which would be the greater one between a negative value and a positive distance value
    heuristics_matrix = torch.clamp(heuristics_matrix, min=0)
    
    return heuristics_matrix