import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total demand
    total_demand = demands.sum()
    
    # Calculate the average demand per customer
    average_demand = total_demand / demands.size(0)
    
    # Calculate the heuristics based on the average demand
    # Promising edges will have a positive value, undesirable edges will have a negative value
    heuristics = (distance_matrix < average_demand).float() * (distance_matrix > 0).float()
    
    return heuristics