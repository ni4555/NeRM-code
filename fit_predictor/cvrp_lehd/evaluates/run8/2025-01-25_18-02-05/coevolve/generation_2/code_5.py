import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize the demands vector by the total vehicle capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity
    
    # Calculate the initial heuristics based on the negative demands (undesirable edges)
    # and the negative distance matrix (to avoid shorter distances)
    initial_heuristics = -normalized_demands - distance_matrix
    
    # The heuristic matrix should be non-negative; thus, clip the values above 0
    heuristics = torch.clamp(initial_heuristics, min=0)
    
    return heuristics