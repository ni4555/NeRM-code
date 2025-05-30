import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total demand
    total_demand = demands.sum()
    
    # Calculate the average demand per customer
    average_demand = total_demand / demands.size(0)
    
    # Calculate the heuristics based on the difference between the average demand and each customer's demand
    heuristics = average_demand - demands
    
    # Normalize the heuristics by the distance matrix
    heuristics = heuristics * distance_matrix
    
    # Ensure that the heuristics are within the range of the distance matrix
    heuristics = torch.clamp(heuristics, min=0, max=distance_matrix.max())
    
    return heuristics