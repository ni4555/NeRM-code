import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize the distance matrix to be between 0 and 1
    distance_matrix = (distance_matrix - distance_matrix.min()) / (distance_matrix.max() - distance_matrix.min())
    
    # Calculate the total distance as a negative heuristic for each edge
    total_distance = distance_matrix.sum(dim=1, keepdim=True)
    
    # Calculate the negative demand as a negative heuristic for each edge
    negative_demand = -demands
    
    # Combine the total distance and negative demand heuristics
    combined_heuristics = total_distance + negative_demand
    
    # Return the heuristics matrix
    return combined_heuristics