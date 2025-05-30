import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize demands by the total vehicle capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity
    
    # Calculate the inverse distance heuristic
    inverse_distances = 1 / (distance_matrix + 1e-8)  # Adding a small constant to avoid division by zero
    
    # Calculate the load balancing heuristic
    load_balancing = (normalized_demands - normalized_demands.mean()) ** 2
    
    # Combine the heuristics
    heuristics = inverse_distances * load_balancing
    
    return heuristics