import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize distance matrix
    min_distance = distance_matrix.min()
    max_distance = distance_matrix.max()
    normalized_distance_matrix = (distance_matrix - min_distance) / (max_distance - min_distance)
    
    # Normalize demands
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity
    
    # Calculate potential values for explicit depot handling
    depot_potential = demands.sum()  # This can be adjusted based on specific problem characteristics
    
    # Calculate heuristic values for each edge
    # Heuristic approach: Use a combination of normalized demand and normalized distance
    heuristic_values = (1 - normalized_distance_matrix) * normalized_demands
    
    # Add explicit depot handling potential
    heuristic_values += depot_potential
    
    return heuristic_values