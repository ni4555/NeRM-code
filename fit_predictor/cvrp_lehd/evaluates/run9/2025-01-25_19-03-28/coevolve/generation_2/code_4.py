import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total demand normalized by the total vehicle capacity
    total_capacity = demands.sum()
    
    # Calculate the normalized demand for each customer
    normalized_demands = demands / total_capacity
    
    # Calculate the heuristic values based on the normalized demand and distance
    # Here we use a simple heuristic that encourages visiting customers with lower normalized demand
    # and closer to the depot (i.e., smaller distance)
    heuristics = -normalized_demands * distance_matrix
    
    # The heuristic values are negative; we want to promote positive values, so we take the absolute value
    # and then we subtract to promote larger values for promising edges
    heuristics = -torch.abs(heuristics)
    
    return heuristics