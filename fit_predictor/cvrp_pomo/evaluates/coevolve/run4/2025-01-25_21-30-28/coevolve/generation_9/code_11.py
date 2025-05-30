import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the inverse of the distance matrix (1 / distance)
    inverse_distance_matrix = torch.reciprocal(distance_matrix)
    
    # Normalize the demands by the sum of all demands
    normalized_demands = demands / demands.sum()
    
    # Calculate the demand weighted distance matrix
    demand_weighted_distance_matrix = distance_matrix * normalized_demands[:, None]
    
    # Combine the inverse distance and demand weighted distance matrices
    combined_heuristic_matrix = -inverse_distance_matrix + demand_weighted_distance_matrix
    
    return combined_heuristic_matrix