import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    vehicle_capacity = demands.sum()
    
    # 1. Precise Demand Handling
    # Normalize demand to reflect the total demand of nodes on potential routes
    normalized_demands = demands / vehicle_capacity
    
    # 2. Capacity Constraint Prioritization
    # Create an edge feasibility mask
    edge_capacity_mask = torch.zeros_like(distance_matrix)
    for i in range(1, n):
        edge_capacity_mask[i, i] = demands[i]  # Edge to itself
    edge_capacity_mask[0, 1:] = demands[1:]  # Edges from depot to customers
    edge_capacity_mask[1:, 0] = demands[1:]  # Edges from customers to depot
    
    # 3. Clear Edge Evaluation
    # Define an evaluation method for edges
    edge_evaluation = -normalized_demands * distance_matrix
    
    # 4. Optimization Strategies
    # Prioritize edges based on their evaluation and capacity constraints
    edge_priority = edge_evaluation + edge_capacity_mask
    
    # Ensure the heuristic values are within the specified range
    edge_priority = torch.clamp(edge_priority, min=-1.0, max=1.0)
    
    return edge_priority