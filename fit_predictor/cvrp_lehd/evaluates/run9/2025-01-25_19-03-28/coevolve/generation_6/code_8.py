import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize demands to be between 0 and 1
    normalized_demands = demands / demands.sum()
    
    # Calculate the load on the vehicle for each customer
    load_on_vehicle = distance_matrix * normalized_demands
    
    # Calculate the total load if we start from the depot
    total_load = load_on_vehicle.sum(dim=1)
    
    # Create a penalty for heavily loaded customers
    load_penalty = torch.where(total_load > 1, torch.ones_like(total_load) * 1000, torch.zeros_like(total_load))
    
    # Calculate the distance to the nearest neighbor heuristic (Manhattan distance)
    heuristic_distance = torch.abs(distance_matrix.sum(dim=1))
    
    # Combine the penalties and the heuristic distance into a single heuristic matrix
    heuristics = load_penalty + heuristic_distance
    
    return heuristics