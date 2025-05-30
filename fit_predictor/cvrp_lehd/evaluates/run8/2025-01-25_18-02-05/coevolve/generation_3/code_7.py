import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total vehicle capacity
    total_capacity = demands.sum()
    
    # Normalize the demands to get the fraction of capacity for each customer
    normalized_demands = demands / total_capacity
    
    # Compute the heuristic values using a simple heuristic where we prioritize edges
    # based on the normalized demand of the customer at the end of the edge.
    # We subtract the demand from 1 to get negative values for higher demand customers.
    # This encourages the algorithm to visit higher demand customers earlier.
    heuristics = 1 - normalized_demands

    return heuristics