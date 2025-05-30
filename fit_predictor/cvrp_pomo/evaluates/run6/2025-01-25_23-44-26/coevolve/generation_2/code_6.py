import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.size(0)
    vehicle_capacity = demands.sum() / demands.size(0)
    
    # Normalize demands by vehicle capacity
    normalized_demands = demands / vehicle_capacity
    
    # Calculate the heuristics based on normalized demands and distances
    heuristics = distance_matrix - normalized_demands.unsqueeze(1)
    
    # Apply a penalty for undesirable edges (e.g., negative values)
    penalty_threshold = -1e-6
    undesirable_edges = heuristics < penalty_threshold
    heuristics[undesirable_edges] = penalty_threshold
    
    return heuristics