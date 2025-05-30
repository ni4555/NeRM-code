import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    # Normalize the distance matrix
    distance_matrix = distance_matrix / distance_matrix.max()
    
    # Calculate the normalized demand
    normalized_demands = demands / demands.sum()
    
    # Demand penalty function: scale the cost of assigning high-demand customers
    demand_penalty = normalized_demands * 0.1  # Example scaling factor
    
    # Calculate the inverse distance heuristic
    inverse_distance = 1 / distance_matrix
    
    # Combine the inverse distance heuristic and the demand penalty
    heuristics = inverse_distance - demand_penalty
    
    # Ensure that the heuristics matrix has negative values for undesirable edges
    heuristics = heuristics - heuristics.max() - 1
    
    return heuristics