import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    total_capacity = demands.sum()
    
    # Precise Demand Handling
    cumulative_demand = torch.cumsum(demands, dim=0)
    
    # Capacity Constraint Prioritization
    edge_capacity_mask = (distance_matrix != 0) * (cumulative_demand[:, None] + demands[None, :] <= total_capacity)
    
    # Clear Edge Evaluation
    edge_evaluation = -distance_matrix  # Negative distance as a heuristic value for edge evaluation
    
    # Optimization Strategies
    # Combine the capacity mask and the edge evaluation to get the final heuristic values
    heuristic_values = edge_capacity_mask * edge_evaluation
    
    return heuristic_values