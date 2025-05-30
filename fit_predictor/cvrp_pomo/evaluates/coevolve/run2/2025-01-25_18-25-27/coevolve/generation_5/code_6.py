import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Ensure that the demands vector is normalized by the total vehicle capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity

    # Calculate the heuristic value for each edge
    # The heuristic is a combination of the normalized demand and the inverse of the distance
    # This is a simple heuristic that assumes shorter distances are more desirable
    # and that the demand per unit distance is a factor in desirability.
    heuristics = normalized_demands.unsqueeze(1) * distance_matrix.unsqueeze(0) + (1 / distance_matrix)
    
    # The resulting heuristics matrix will have the same shape as the distance matrix
    # and will contain positive values for promising edges and negative values for undesirable ones.
    
    return heuristics