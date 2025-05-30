import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Inverse distance heuristic: Promote edges with shorter distances
    inverse_distance = -distance_matrix
    
    # Demand normalization heuristic: Normalize demands by total capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity
    
    # Combine heuristics: Multiply the inverse distance by normalized demand
    combined_heuristic = inverse_distance * normalized_demands
    
    return combined_heuristic