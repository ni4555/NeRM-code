import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the sum of demands to normalize
    total_demand = demands.sum()
    
    # Normalize the demands
    normalized_demands = demands / total_demand
    
    # Create a matrix of 1s to represent all possible edges
    edge_matrix = torch.ones_like(distance_matrix)
    
    # Subtract the normalized demand from each edge to get a heuristic value
    # This heuristic value is negative if the edge is not promising and positive if it is
    heuristics = edge_matrix - normalized_demands[:, None] - normalized_demands[None, :]
    
    # Replace infinite distances with 0s to avoid division by zero in normalization
    heuristics = torch.clamp(heuristics, min=0)
    
    return heuristics