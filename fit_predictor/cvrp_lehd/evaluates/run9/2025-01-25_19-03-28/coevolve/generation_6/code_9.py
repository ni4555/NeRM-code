import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total vehicle capacity
    vehicle_capacity = demands.sum()
    
    # Normalize customer demands to the total vehicle capacity
    normalized_demands = demands / vehicle_capacity
    
    # Initialize the heuristics matrix with zeros
    heuristics_matrix = torch.zeros_like(distance_matrix)
    
    # Calculate the potential cost of visiting each customer
    # This is a simplistic heuristic where we assume the cost is inversely proportional to the demand
    cost_potential = 1 / (normalized_demands + 1e-10)  # Add a small value to avoid division by zero
    
    # Apply the cost potential to the distance matrix
    heuristics_matrix = cost_potential * distance_matrix
    
    return heuristics_matrix