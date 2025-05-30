import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize distance matrix to ensure it's non-negative and scale it up
    normalized_distance_matrix = (distance_matrix - torch.min(distance_matrix)) / (torch.max(distance_matrix) - torch.min(distance_matrix))
    
    # Normalize demands to sum to 1 for easier load balance
    normalized_demands = demands / torch.sum(demands)
    
    # Compute the heuristics based on the distance matrix and normalized demands
    # Here we use a simple heuristic that promotes edges with low distance and high demand
    heuristics = -normalized_distance_matrix * demands
    
    # Optionally, add more sophisticated heuristics here
    
    return heuristics