import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total vehicle capacity (assuming all demands are normalized)
    total_capacity = demands.sum()
    
    # Calculate the demand contribution of each edge (i.e., the difference in demand
    # between the two nodes connected by the edge)
    demand_difference = demands.unsqueeze(1) - demands.unsqueeze(0)
    
    # Calculate the negative sum of the demand differences (to make higher demand
    # differences more desirable)
    negative_demand_sum = -demand_difference.sum(dim=2)
    
    # Calculate the distance penalty for each edge
    distance_penalty = distance_matrix
    
    # Combine the demand and distance contributions into a single heuristic value
    heuristic_values = negative_demand_sum + distance_penalty
    
    # Normalize the heuristic values to ensure they are within the range of the
    # original distance matrix
    min_val = heuristic_values.min()
    max_val = heuristic_values.max()
    heuristic_values = (heuristic_values - min_val) / (max_val - min_val)
    
    return heuristic_values