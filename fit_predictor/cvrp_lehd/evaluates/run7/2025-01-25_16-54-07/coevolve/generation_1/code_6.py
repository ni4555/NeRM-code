import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the difference in demands between each pair of nodes
    demand_diff = demands.unsqueeze(1) - demands.unsqueeze(0)
    
    # Apply a threshold to the demand difference to identify promising edges
    # This threshold could be determined by domain knowledge or experimentation
    threshold = 0.1
    promising_demand_diff = torch.where(demand_diff.abs() > threshold, 1, 0)
    
    # Use the distance matrix to penalize longer distances
    # This could be a linear or exponential function of the distance
    # Here we use a simple linear function as an example
    distance_penalty = distance_matrix * 0.1
    
    # Combine the demand difference and distance penalty to get the heuristic values
    heuristics = promising_demand_diff - distance_penalty
    
    return heuristics