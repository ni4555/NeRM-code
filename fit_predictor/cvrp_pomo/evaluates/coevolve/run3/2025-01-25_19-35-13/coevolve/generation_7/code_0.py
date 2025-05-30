import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Ensure the demands are normalized by the total vehicle capacity
    vehicle_capacity = demands.sum()
    normalized_demands = demands / vehicle_capacity

    # Calculate the potential function
    # Here, we assume a simple potential function based on the sum of normalized demands
    # and the distance, with a penalty for large demands and a discount for distance
    # Adjust the parameters according to the problem's characteristics
    demand_weight = 0.5
    distance_weight = 0.3
    road_quality_penalty = 0.2  # Example of incorporating road quality into the potential

    # Calculate the sum of normalized demands for each edge
    edge_demand_sum = torch.matmul(normalized_demands.unsqueeze(1), normalized_demands.unsqueeze(0))
    
    # Incorporate distance and road quality into the potential function
    potential = (edge_demand_sum * demand_weight) - (distance_matrix * distance_weight)
    
    # Introduce a penalty for high demands
    high_demand_penalty = torch.clamp((1 - (edge_demand_sum * demand_weight)), min=0, max=1)
    potential = potential + (high_demand_penalty * road_quality_penalty)
    
    # Ensure no division by zero errors
    potential = torch.clamp(potential, min=0)
    
    return potential