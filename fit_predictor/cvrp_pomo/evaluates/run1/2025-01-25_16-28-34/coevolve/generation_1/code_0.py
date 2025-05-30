import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Assuming that the distance_matrix and demands are 1-D tensors after removing the depot node
    # Normalize the demands to have a sum of 1 for the purpose of heuristics
    demands_normalized = demands / demands.sum()
    
    # Calculate the heuristic as a product of normalized demand and distance
    # We use the square of the distance to emphasize shorter paths
    heuristics = (demands_normalized.unsqueeze(1) * distance_matrix.unsqueeze(0)) ** 2
    
    # Negative values indicate undesirable edges
    # For simplicity, we assume that any edge to a customer with zero demand is undesirable
    heuristics[torch.nonzero(demands == 0)] *= -1
    
    return heuristics