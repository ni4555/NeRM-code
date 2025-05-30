import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Ensure that the demands are normalized
    total_capacity = demands.sum()
    demands = demands / total_capacity
    
    # Initialize the heuristics matrix with zeros
    heuristics = torch.zeros_like(distance_matrix)
    
    # Calculate the cost of visiting each customer from the depot
    # The cost is a combination of the inverse of the demand and the distance to the customer
    heuristics[:, 1:] = (1 / demands[1:]) * distance_matrix[:, 1:]
    heuristics[0, 1:] = -torch.inf  # The depot has no cost to visit itself
    
    # Add a term to penalize high demands (if any)
    heuristics[:, 1:] += demands[1:] * 1000  # This is a hyperparameter that can be adjusted
    
    return heuristics