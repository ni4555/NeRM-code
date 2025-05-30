import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the sum of demands to normalize
    total_demand = demands.sum().item()
    
    # Normalize the demands
    normalized_demands = demands / total_demand
    
    # Calculate the service time for each customer
    service_time = 1.0  # Assuming each customer service takes 1 unit of time
    
    # Initialize the heuristic matrix with zeros
    heuristic_matrix = torch.zeros_like(distance_matrix)
    
    # Incorporate distance as a negative heuristic (shorter distances are better)
    heuristic_matrix += -distance_matrix
    
    # Incorporate demand as a positive heuristic (higher demand is better)
    # Here we use a demand scaling factor to balance the influence of demand
    demand_scaling_factor = 0.5
    heuristic_matrix += demand_scaling_factor * (normalized_demands * service_time)
    
    # Incorporate a balance between distance and demand
    # For example, a balance factor of 0.5 between distance and demand
    balance_factor = 0.5
    heuristic_matrix = balance_factor * heuristic_matrix
    
    return heuristic_matrix