import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize the demands to be within [0, 1]
    demand_ratio = demands / demands.sum()
    
    # Calculate the sum of distances multiplied by the demand ratios
    # This gives a measure of importance for each edge
    importance_matrix = distance_matrix * demand_ratio[None, :] * demand_ratio[:, None]
    
    # Subtract the importance matrix from the distance matrix to create a heuristics matrix
    # Negative values will indicate undesirable edges (high cost with little demand)
    heuristics_matrix = distance_matrix - importance_matrix
    
    # To ensure that the heuristics matrix has a consistent shape with the distance matrix
    return heuristics_matrix