import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Ensure that demands are normalized by the total vehicle capacity
    vehicle_capacity = demands[0]  # Assuming the demand at the depot node is the total capacity
    demands /= vehicle_capacity
    
    # Initialize a matrix of zeros with the same shape as the distance matrix
    heuristic_matrix = torch.zeros_like(distance_matrix)
    
    # Normalize the distance matrix
    normalized_distance_matrix = distance_matrix / vehicle_capacity
    
    # Incorporate demand into the heuristic
    demand_weighted_distance = normalized_distance_matrix * demands
    
    # Introduce a penalty for exceeding the demand at each node
    penalty_excess_demand = (demands - 1) * 1000  # Arbitrary large value for excess demand
    
    # Create a matrix of penalties for edges leading to nodes with excess demand
    penalty_matrix = penalty_excess_demand[torch.arange(distance_matrix.shape[0]), torch.arange(distance_matrix.shape[0])]
    
    # Combine the demand-weighted distances with the penalties
    heuristic_matrix = demand_weighted_distance - penalty_matrix
    
    return heuristic_matrix