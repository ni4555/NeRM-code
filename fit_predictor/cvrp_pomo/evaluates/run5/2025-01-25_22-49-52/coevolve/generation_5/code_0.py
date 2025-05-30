import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the demand per unit distance heuristic
    demand_per_unit_distance = demands / distance_matrix
    
    # Normalize the demand per unit distance by the maximum demand per unit distance
    # This normalization helps in consistent scaling
    max_demand_per_unit_distance = demand_per_unit_distance.max()
    normalized_demand_per_unit_distance = demand_per_unit_distance / max_demand_per_unit_distance
    
    # Calculate the heuristic as the negative of the normalized demand per unit distance
    # Negative values represent undesirable edges (high demand)
    # Positive values represent promising edges (low demand)
    heuristic = -normalized_demand_per_unit_distance
    
    return heuristic