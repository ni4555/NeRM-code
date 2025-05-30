import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total vehicle capacity
    total_capacity = demands.sum()
    
    # Calculate the normalized demands
    normalized_demands = demands / total_capacity
    
    # Calculate the heuristic values for each edge
    # The heuristic is based on the negative of the demand multiplied by the distance
    # This encourages routes with lower demand and shorter distances
    heuristics = -normalized_demands.unsqueeze(1) * distance_matrix
    
    # To ensure that the heuristic values are negative for undesirable edges,
    # we add a very large number to the edges that are not part of the matrix (i.e., self-loops)
    # and subtract it from the others
    heuristics += torch.max(heuristics, distance_matrix.new_zeros_like(heuristics) + 1e10)
    
    return heuristics