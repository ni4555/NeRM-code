import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Ensure that demands are normalized by dividing by the total capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity
    
    # Calculate the potential negative heuristic for each edge based on customer demand
    demand_based_heuristic = -normalized_demands.repeat(len(distance_matrix))
    
    # Optionally, add more complex heuristics here, for instance:
    # - A distance-based heuristic could be subtracted or added
    # - Interaction terms between demands could be considered
    # - ... etc.
    
    # For demonstration purposes, we'll just return the demand-based heuristic
    return demand_based_heuristic