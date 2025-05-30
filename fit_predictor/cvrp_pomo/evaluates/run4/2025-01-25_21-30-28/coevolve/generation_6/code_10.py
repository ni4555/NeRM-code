import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    total_capacity = demands.sum()
    demands_normalized = demands / total_capacity
    
    # Inverse distance heuristic
    inverse_distance = 1 / (distance_matrix + 1e-8)  # Adding a small constant to avoid division by zero
    
    # Normalization heuristic
    normalization = demands_normalized / demands_normalized.sum()
    
    # Combine heuristics, giving a higher weight to inverse distance
    combined_heuristics = inverse_distance * 0.7 + normalization * 0.3
    
    # Make some edges undesirable by subtracting a large value
    undesirable_edges = torch.full_like(combined_heuristics, -1e6)
    
    # Set diagonal to zero as no distance to itself
    torch.fill_diagonal_(combined_heuristics, 0)
    
    # Set diagonal of undesirable edges to zero
    torch.fill_diagonal_(undesirable_edges, 0)
    
    # Return the result where desirable edges are positive and undesirable are negative
    return combined_heuristics + undesirable_edges