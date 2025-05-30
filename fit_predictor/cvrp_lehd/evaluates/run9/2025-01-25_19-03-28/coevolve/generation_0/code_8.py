import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total demand normalized by the vehicle capacity
    vehicle_capacity = demands[0]  # Assuming the first customer demand is the vehicle capacity
    total_demand = torch.sum(demands[1:])  # Sum of all customer demands
    
    # Calculate the difference between vehicle capacity and total demand
    capacity_difference = vehicle_capacity - total_demand
    
    # Create a matrix of edge costs
    # Negative values for edges where the demand is greater than the vehicle capacity
    # Positive values for other edges
    edge_costs = -torch.abs(demands[1:] - vehicle_capacity)
    
    # Normalize the edge costs by the capacity difference to maintain a similar scale
    edge_costs = edge_costs / capacity_difference
    
    # Add the depot costs (0 for depot to itself)
    edge_costs = torch.cat((torch.zeros(1), edge_costs), dim=0)
    edge_costs = torch.cat((edge_costs, torch.zeros(1)), dim=0)
    
    return edge_costs