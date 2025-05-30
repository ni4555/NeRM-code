import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the sum of demands to normalize them
    total_capacity = demands.sum()
    
    # Normalize the demands
    normalized_demands = demands / total_capacity
    
    # Calculate the sum of distances for each edge and subtract the demand of the destination node
    edge_sums = distance_matrix.sum(dim=1) - demands
    
    # Use the normalized demand to weigh the edge sums
    weighted_edge_sums = edge_sums * normalized_demands
    
    # The heuristic value is the negative of the weighted edge sums
    # Negative values indicate undesirable edges, positive values indicate promising ones
    heuristics = -weighted_edge_sums
    
    return heuristics