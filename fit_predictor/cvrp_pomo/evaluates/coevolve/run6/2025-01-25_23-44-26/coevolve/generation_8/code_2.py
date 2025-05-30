import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize customer demands by the total vehicle capacity (assuming it's 1 for simplicity)
    normalized_demands = demands / demands.sum()
    
    # Calculate the inverse distance heuristic
    inverse_distance = 1 / distance_matrix
    
    # Incorporate load balancing by adjusting the inverse distance based on demand
    load_balancing_factor = torch.clamp(inverse_distance * normalized_demands, min=0, max=1)
    
    # Combine the heuristics
    combined_heuristics = load_balancing_factor
    
    return combined_heuristics