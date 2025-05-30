import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize demands to the range [0, 1]
    demands_normalized = demands / demands.sum()
    
    # Calculate the difference in demand between each customer and the average demand
    demand_diff = demands_normalized - demands_normalized.mean()
    
    # Use the difference in demand as a heuristic
    heuristics = demand_diff * distance_matrix
    
    # Ensure the heuristic values are negative for undesirable edges and positive for promising ones
    heuristics[distance_matrix == 0] = 0  # Set the diagonal to zero, as the depot should not be included in the solution
    heuristics[heuristics < 0] = 0  # Set negative values to zero
    
    return heuristics