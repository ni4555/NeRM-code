import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Assuming the demands vector includes the depot node's demand which is zero, we don't need to adjust for it.
    
    # Normalize the distance matrix by dividing each element by the vehicle capacity.
    # The capacity is normalized as a single value so it can be applied to the entire distance matrix.
    normalized_distance_matrix = distance_matrix / demands[0]
    
    # Calculate the difference between the normalized demands and 1, which represents the penalty for high demand.
    # We multiply by -1 because we want to use negative values for undesirable edges.
    demand_penalty = -torch.abs(1 - demands[1:]) / demands[1:]
    
    # Combine the normalized distance with the demand penalty using element-wise multiplication.
    heuristics = normalized_distance_matrix * demand_penalty
    
    return heuristics