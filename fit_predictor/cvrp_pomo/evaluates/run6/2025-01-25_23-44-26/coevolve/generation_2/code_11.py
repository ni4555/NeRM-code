import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total vehicle capacity
    total_capacity = demands.sum()
    
    # Normalize customer demands to the total vehicle capacity
    normalized_demands = demands / total_capacity
    
    # Calculate the total distance for each node to all other nodes
    total_distances = distance_matrix.sum(dim=1)
    
    # Initialize the heuristic matrix with negative values
    heuristic_matrix = -torch.ones_like(distance_matrix)
    
    # Adjust the heuristic values based on normalized demands and distances
    # For each node, calculate the heuristic as the demand contribution minus the distance
    heuristic_matrix = normalized_demands.unsqueeze(1) - distance_matrix
    
    # Replace negative values with zeros to discourage undesirable edges
    heuristic_matrix = torch.clamp(heuristic_matrix, min=0)
    
    return heuristic_matrix