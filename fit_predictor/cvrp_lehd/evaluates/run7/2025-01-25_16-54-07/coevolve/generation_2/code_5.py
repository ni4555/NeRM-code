import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize demands to the range [0, 1]
    normalized_demands = demands / demands.sum()
    
    # Calculate the potential of each edge based on the difference in demands
    # between the two nodes and the distance between them
    edge_potential = normalized_demands.unsqueeze(1) - normalized_demands.unsqueeze(0)
    edge_potential = (edge_potential * distance_matrix).sum(dim=2)
    
    # Introduce a penalty for longer distances
    distance_penalty = -distance_matrix
    
    # Combine the potential and penalty to form the heuristic
    heuristics = edge_potential + distance_penalty
    
    return heuristics