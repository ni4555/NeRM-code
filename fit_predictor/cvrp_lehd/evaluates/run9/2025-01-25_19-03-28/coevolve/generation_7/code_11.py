import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Ensure demands are normalized
    demands = demands / demands.sum()
    
    # Initialize heuristics matrix with zeros
    n = distance_matrix.shape[0]
    heuristics = torch.zeros_like(distance_matrix)
    
    # Compute heuristic values for each edge
    # Negative values for edges leading to high demands and positive values for shorter distances
    heuristics[distance_matrix > 0] = -demands[distance_matrix > 0]
    
    # Adjust heuristic values for edges leading to the depot (customers)
    depot_mask = (distance_matrix == 0) & (demands != 0)
    heuristics[depot_mask] = demands[depot_mask] / distance_matrix[depot_mask]
    
    return heuristics