import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total demand normalized by vehicle capacity
    total_demand = demands.sum()
    
    # Calculate the sum of demands of all nodes excluding the depot
    node_demand_sum = demands[1:]
    
    # Calculate the sum of demands for each row (origin node)
    row_demand_sum = demands.unsqueeze(1).sum(0)
    
    # Calculate the sum of demands for each column (destination node)
    column_demand_sum = demands.unsqueeze(0).sum(0)
    
    # Create a tensor to hold the heuristic values initialized to a default negative value
    heuristic_values = torch.full_like(distance_matrix, fill_value=-1e5)
    
    # Update the heuristic values for promising edges (positive values)
    # 1. Promote edges between nodes with low remaining demand and origin nodes with low total demand
    heuristic_values[1:, 1:] += (row_demand_sum[1:] < total_demand / 2) * (node_demand_sum < total_demand / 2)
    
    # 2. Promote edges between nodes with high remaining demand and origin nodes with high total demand
    heuristic_values[1:, 1:] += (row_demand_sum[1:] > total_demand / 2) * (node_demand_sum > total_demand / 2)
    
    # Return the heuristic values
    return heuristic_values