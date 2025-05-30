import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize demands by total vehicle capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity
    
    # Compute the potential of each edge based on distance and demand
    potential = distance_matrix - normalized_demands.unsqueeze(1) * distance_matrix
    
    # Set a small value for undesirable edges to ensure they are not favored
    undesirable_threshold = -1e-5
    undesirable_edges = (potential < undesirable_threshold)
    potential[undesirable_edges] = undesirable_threshold
    
    return potential