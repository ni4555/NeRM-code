import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Ensure demands are normalized
    demands = demands / demands.sum()
    
    # Initialize a tensor to store heuristics values with the same shape as distance_matrix
    heuristics = torch.zeros_like(distance_matrix)
    
    # Calculate the heuristic value for each edge
    # We use a simple heuristic that considers the inverse of the distance (shorter distances are better)
    # and the normalized demand of the customer (customers with higher demand might be more urgent).
    heuristics = (1.0 / distance_matrix) * demands
    
    # We could introduce additional factors such as the capacity of the vehicles or constraints
    # Here, we assume that the demand normalization is enough to reflect the urgency of the customer
    
    return heuristics