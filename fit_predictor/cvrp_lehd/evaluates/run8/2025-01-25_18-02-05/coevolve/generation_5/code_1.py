import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the negative of the distance matrix for the heuristic values
    heuristic_matrix = -distance_matrix
    
    # Normalize the heuristic matrix based on customer demands
    # This can be a simple approach such as the ratio of demand to distance
    # For example, we can use the ratio of the demand to the distance to the next node
    # as a heuristic score. Here we assume that the demand is normalized by the total vehicle capacity.
    # Thus, we multiply the demand by the distance to give a heuristic score.
    heuristic_matrix = heuristic_matrix * demands
    
    return heuristic_matrix