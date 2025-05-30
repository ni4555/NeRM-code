import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the inverse distance heuristic
    inv_distance = 1 / (distance_matrix + 1e-6)  # Adding a small constant to avoid division by zero
    
    # Normalize the customer demands
    total_demand = demands.sum()
    normalized_demands = demands / total_demand
    
    # Combine the inverse distance and normalized demands using a weighted sum
    # The weights can be adjusted to favor either distance or demand
    weight_distance = 0.5
    weight_demand = 0.5
    heuristic_matrix = weight_distance * inv_distance + weight_demand * normalized_demands
    
    return heuristic_matrix