import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    vehicle_capacity = demands.sum()
    
    # 1. Precise Demand Handling
    # Normalize demands
    normalized_demands = demands / vehicle_capacity
    
    # Calculate cumulative demand for each edge
    cumulative_demand = torch.cumsum(normalized_demands, dim=0)
    
    # 2. Capacity Constraint Prioritization
    # Create an edge feasibility mask
    edge_capacity_mask = distance_matrix < vehicle_capacity
    
    # 3. Clear Edge Evaluation
    # Evaluate edges based on distance and cumulative demand
    edge_evaluation = -distance_matrix + cumulative_demand
    
    # Apply edge feasibility mask
    edge_evaluation = edge_evaluation * edge_capacity_mask
    
    return edge_evaluation