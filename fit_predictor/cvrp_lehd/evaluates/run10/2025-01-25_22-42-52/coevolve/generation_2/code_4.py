import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Ensure demands are normalized
    demand_sum = demands.sum()
    normalized_demands = demands / demand_sum
    
    # Calculate the difference between demands and normalized demands
    demand_diff = demands - normalized_demands
    
    # Compute a basic heuristic based on the demand difference
    basic_heuristic = demand_diff
    
    # Add a penalty for edges that are far away (this discourages longer distances)
    distance_penalty = distance_matrix * demand_diff
    
    # Combine the basic heuristic and distance penalty to form the heuristic matrix
    heuristics = basic_heuristic - distance_penalty
    
    # Cap the heuristic values to avoid extreme negative values (edges to be avoided)
    heuristics = torch.clamp(heuristics, min=-1e5)
    
    return heuristics