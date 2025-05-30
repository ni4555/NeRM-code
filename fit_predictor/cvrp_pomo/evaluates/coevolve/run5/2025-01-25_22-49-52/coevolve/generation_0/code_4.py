import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the sum of demands to normalize
    total_demand = demands.sum()
    
    # Normalize the demands by the total vehicle capacity
    normalized_demands = demands / total_demand
    
    # Calculate the potential cost of each edge (distance * normalized demand)
    potential_costs = distance_matrix * normalized_demands
    
    # Apply a threshold to promote edges with lower potential costs
    threshold = 0.5  # This threshold can be adjusted
    heuristics = torch.where(potential_costs < threshold, potential_costs, -torch.inf)
    
    return heuristics