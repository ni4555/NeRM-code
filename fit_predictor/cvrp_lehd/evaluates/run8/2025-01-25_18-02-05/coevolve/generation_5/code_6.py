import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the cumulative sum of demands to the right of each customer
    cumulative_demand = demands.cumsum(0)
    
    # Calculate the cumulative sum of demands to the left of each customer
    cumulative_demand_right = cumulative_demand.flip(0)
    
    # Calculate the total demand up to each customer
    total_demand_up_to = cumulative_demand + cumulative_demand_right
    
    # Calculate the heuristics based on the difference in total demand up to each customer
    heuristics = (total_demand_up_to - distance_matrix).cumsum(1)
    
    # Normalize the heuristics to the range of [0, 1]
    heuristics /= heuristics.max()
    
    return heuristics