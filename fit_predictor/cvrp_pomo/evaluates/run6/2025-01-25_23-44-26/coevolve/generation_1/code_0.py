import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the cumulative demand for each node
    cumulative_demand = torch.cumsum(demands, dim=0)
    
    # Calculate the difference in cumulative demand from each node to the next
    demand_diff = cumulative_demand[1:] - cumulative_demand[:-1]
    
    # Calculate the heuristic value as the negative of the demand difference
    # This encourages routes that balance the load
    heuristic_matrix = -torch.abs(demand_diff)
    
    # Adjust the heuristic matrix based on the distance matrix
    # Here we are simply using the distance matrix to add a penalty for longer distances
    # This can be adjusted or removed based on the specific requirements of the problem
    heuristic_matrix += distance_matrix
    
    return heuristic_matrix