import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total demand
    total_demand = demands.sum()
    
    # Calculate the minimum demand among all customers
    min_demand = demands.min()
    
    # Calculate the maximum demand among all customers
    max_demand = demands.max()
    
    # Calculate the average demand among all customers
    average_demand = demands.mean()
    
    # Calculate the normalized demands
    normalized_demands = demands / total_demand
    
    # Calculate the number of customers
    num_customers = demands.size(0)
    
    # Create a vector to store the heuristics
    heuristics = torch.zeros_like(distance_matrix)
    
    # Heuristic 1: Negative values for edges with high demand
    heuristics[distance_matrix > max_demand] = -1.0
    
    # Heuristic 2: Positive values for edges with average demand
    heuristics[distance_matrix == average_demand] = 1.0
    
    # Heuristic 3: Negative values for edges with low demand
    heuristics[distance_matrix < min_demand] = -1.0
    
    # Heuristic 4: Positive values for edges with normalized demand close to 1
    heuristics[(distance_matrix < total_demand) & (normalized_demands > 0.8)] = 1.0
    
    # Heuristic 5: Negative values for edges with normalized demand close to 0
    heuristics[(distance_matrix < total_demand) & (normalized_demands < 0.2)] = -1.0
    
    return heuristics