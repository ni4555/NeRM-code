import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the demand per unit distance for each edge
    demand_per_distance = demands / distance_matrix
    
    # Normalize by the maximum demand per distance to ensure positive values
    max_demand_per_distance = torch.max(demand_per_distance)
    normalized_demand_per_distance = demand_per_distance / max_demand_per_distance
    
    # The heuristics are based on the normalized demand per distance
    # Higher demand per distance suggests a more promising edge
    heuristics = normalized_demand_per_distance
    
    return heuristics