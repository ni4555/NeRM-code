import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the sum of distances for each row (each route from the depot)
    sum_distances = distance_matrix.sum(dim=1)
    
    # Calculate the total demand for each route
    total_demand = demands.sum()
    
    # Normalize the sum of distances by the total demand to get a relative cost
    normalized_costs = sum_distances / total_demand
    
    # Calculate the potential of each edge by considering the balance between cost and demand
    # Promising edges will have lower cost relative to their demand, hence negative values
    heuristics = normalized_costs - demands
    
    return heuristics