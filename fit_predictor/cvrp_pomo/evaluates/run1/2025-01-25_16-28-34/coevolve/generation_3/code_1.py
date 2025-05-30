import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize demand
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity
    
    # Normalize distance matrix by the square root of the demand
    distance_matrix_normalized = distance_matrix / torch.sqrt(normalized_demands)
    
    # Initialize the heuristic matrix with high negative values for undesirable edges
    heuristic_matrix = -torch.ones_like(distance_matrix_normalized)
    
    # Find edges with non-zero demand
    non_zero_demand_edges = distance_matrix_normalized != 0
    
    # Set heuristic values for edges with non-zero demand
    heuristic_matrix[non_zero_demand_edges] = distance_matrix_normalized[non_zero_demand_edges]
    
    return heuristic_matrix