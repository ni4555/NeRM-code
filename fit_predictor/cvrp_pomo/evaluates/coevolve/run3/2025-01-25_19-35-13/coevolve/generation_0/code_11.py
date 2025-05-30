import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the sum of demands to normalize by the total vehicle capacity
    total_demand = demands.sum()
    
    # Normalize demands by the total vehicle capacity
    normalized_demands = demands / total_demand
    
    # Calculate the potential value for each edge based on normalized demand and distance
    potential_value = normalized_demands * distance_matrix
    
    # Introduce a penalty for edges leading to the depot (index 0)
    depot_penalty = torch.full_like(potential_value, -1e6)
    potential_value[distance_matrix == 0] = depot_penalty[distance_matrix == 0]
    
    # Introduce a penalty for edges with high normalized demand
    demand_penalty = torch.full_like(potential_value, -1e6)
    high_demand_edges = normalized_demands > 0.5  # Example threshold
    potential_value[high_demand_edges] = demand_penalty[high_demand_edges]
    
    return potential_value