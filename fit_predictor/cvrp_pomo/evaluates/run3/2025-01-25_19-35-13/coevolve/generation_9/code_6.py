import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.size(0)
    # Initialize the heuristics matrix with large negative values
    heuristics = torch.full((n, n), -float('inf'))
    
    # Set diagonal to zero as it represents distance from a node to itself
    torch.fill_diagonal_(heuristics, 0)
    
    # Calculate the total demand
    total_demand = demands.sum()
    
    # Normalize demands by total capacity
    normalized_demands = demands / total_demand
    
    # Calculate the cost of each edge
    edge_costs = torch.abs(distance_matrix)
    
    # Calculate the heuristics for each edge
    for i in range(n):
        for j in range(n):
            if i != j:
                # Calculate the remaining capacity of the vehicle
                remaining_capacity = 1.0 - demands[i]
                # Calculate the potential of the edge
                potential = edge_costs[i, j] - remaining_capacity * normalized_demands[j]
                heuristics[i, j] = max(heuristics[i, j], potential)
    
    # Set the diagonal back to zero if it was changed
    torch.fill_diagonal_(heuristics, 0)
    
    return heuristics