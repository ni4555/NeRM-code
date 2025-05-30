import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the negative distance matrix as an indication of undesirable edges
    negative_distance_matrix = -distance_matrix
    
    # Normalize the demands vector to be in the range [0, 1] for the purpose of this heuristic
    normalized_demands = demands / demands.sum()
    
    # Generate the heuristics matrix based on negative distances and normalized demands
    heuristics_matrix = negative_distance_matrix + (demands * normalized_demands).unsqueeze(1)
    
    # Replace negative values with zeros, as we are interested in promising edges
    heuristics_matrix = torch.clamp(heuristics_matrix, min=0)
    
    return heuristics_matrix