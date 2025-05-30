import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total vehicle capacity
    total_capacity = demands.sum()
    
    # Normalize demands
    normalized_demands = demands / total_capacity
    
    # Calculate the inverse distance heuristic
    idh = 1 / (distance_matrix + 1e-6)  # Adding a small constant to avoid division by zero
    
    # Initialize the heuristics matrix with the inverse distance heuristic values
    heuristics = idh
    
    # Calculate the penalty function for capacity constraints
    # Here, we assume that the penalty is linearly proportional to the distance from the demand threshold
    # The demand threshold is set to 0.5, meaning the penalty is applied when the demand is 50% or more of the capacity
    demand_threshold = 0.5
    capacity_penalty = (normalized_demands - demand_threshold) * (distance_matrix * 0.1)
    
    # Combine the heuristics with the capacity penalty
    heuristics += capacity_penalty
    
    # Ensure that edges with negative heuristics are set to zero (not desirable)
    heuristics = torch.clamp(heuristics, min=0)
    
    return heuristics