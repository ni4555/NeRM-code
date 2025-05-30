import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    # Normalize demand vector to the range [0, 1]
    normalized_demands = demands / demands.sum()
    
    # Initialize the heuristic matrix with zeros
    heuristic_matrix = torch.zeros_like(distance_matrix)
    
    # Calculate the normalized distance matrix
    normalized_distance_matrix = distance_matrix / distance_matrix.sum(axis=1, keepdim=True)
    
    # Calculate the heuristic for each edge
    # The heuristic is a combination of normalized demand and normalized distance
    heuristic_matrix = normalized_demands.unsqueeze(1) * normalized_distance_matrix
    
    # Add a penalty for edges that go against the direction of demand (i.e., from high to low demand)
    # This encourages load balancing
    penalty = torch.abs(torch.cumsum(normalized_demands, dim=0) - torch.arange(n))
    heuristic_matrix += penalty.unsqueeze(1)
    
    return heuristic_matrix