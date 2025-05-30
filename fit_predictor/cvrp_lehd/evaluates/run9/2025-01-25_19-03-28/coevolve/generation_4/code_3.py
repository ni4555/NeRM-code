import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the maximum distance in the matrix to normalize the heuristic
    max_distance = torch.max(distance_matrix)
    
    # Normalize the distance matrix by dividing by the maximum distance
    normalized_distance_matrix = distance_matrix / max_distance
    
    # Calculate the heuristic value based on the demand and normalized distance
    # We use a simple heuristic: more demanding customers are more likely to be included
    # Negative heuristic for undesirable edges, positive for promising ones
    heuristics = -demands * normalized_distance_matrix
    
    return heuristics