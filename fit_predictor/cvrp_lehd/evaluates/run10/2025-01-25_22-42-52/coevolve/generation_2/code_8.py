import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the demand per unit distance (normalized demand)
    demand_per_unit_distance = demands / distance_matrix
    
    # Calculate the heuristic as the negative of the demand per unit distance
    # (promising edges will have higher demand per unit distance, hence the negative sign)
    heuristic_matrix = -demand_per_unit_distance
    
    # Replace the diagonal values with a large negative value to avoid selecting the depot as a customer
    # (the depot should not be included in the solution)
    torch.fill_diagonal_(heuristic_matrix, -float('inf'))
    
    return heuristic_matrix