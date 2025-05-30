import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total vehicle capacity
    total_capacity = demands.sum()
    
    # Calculate the normalized demands
    normalized_demands = demands / total_capacity
    
    # Calculate the heuristics values
    # For each customer, compute the potential cost of serving it, which is the distance to the customer
    # multiplied by the normalized demand (potential benefit)
    heuristics = -distance_matrix * normalized_demands
    
    # Optionally, you could add more sophisticated heuristics like the savings algorithm,
    # savings = distance_matrix[i][j] - distance_matrix[i][k] - distance_matrix[k][j]
    # for a 3-customer problem with k as the depot.
    
    return heuristics