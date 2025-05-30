import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    # Calculate the total distance from the depot to all customers and back to the depot
    depot_to_customers = distance_matrix[0, 1:] + distance_matrix[1:, 0]
    total_distance = depot_to_customers.sum() + n  # n is the number of customers
    
    # Normalize demands by the total vehicle capacity (assuming capacity is 1 for simplicity)
    normalized_demands = demands / demands.sum()
    
    # Compute the heuristic values as a combination of negative total distance and demand
    heuristics = -total_distance * normalized_demands
    
    return heuristics