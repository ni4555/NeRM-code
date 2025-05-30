import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Inverse Distance Heuristic (IDH)
    # Normalize the distance matrix to get a uniform scale
    distance_matrix = distance_matrix / distance_matrix.max()
    
    # Add a penalty for high demand, using a simple inverse relationship
    demand_penalty = 1 / (demands + 1e-6)  # Add a small constant to avoid division by zero
    
    # Combine the inverse distance and demand penalties
    heuristic_matrix = -distance_matrix + demand_penalty
    
    # Normalize the heuristic matrix to get a uniform scale for edge evaluation
    heuristic_matrix = heuristic_matrix / heuristic_matrix.max()
    
    return heuristic_matrix