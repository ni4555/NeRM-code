import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the demand-to-distance ratio for each edge
    demand_to_distance_ratio = demands / distance_matrix
    
    # Calculate the negative of this ratio to use as a heuristic
    # Negative values indicate undesirable edges (heuristic score is lower)
    heuristic_matrix = -demand_to_distance_ratio
    
    # To ensure that we have a proper heuristic with negative values for undesirable edges,
    # we can set a threshold. This threshold can be chosen based on empirical evidence or
    # some heuristic.
    threshold = torch.min(heuristic_matrix)
    
    # Apply the threshold to ensure all undesirable edges have negative values
    heuristic_matrix = torch.where(heuristic_matrix < threshold, heuristic_matrix, 0)
    
    return heuristic_matrix