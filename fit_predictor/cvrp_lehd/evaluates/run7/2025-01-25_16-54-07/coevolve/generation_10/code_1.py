import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    total_capacity = demands.sum()
    demands_normalized = demands / total_capacity

    # 1. Precise Demand Handling
    # Calculate cumulative demand for each potential route
    cumulative_demand = torch.cumsum(demands_normalized, dim=0)

    # 2. Capacity Constraint Prioritization
    # Create edge feasibility mask based on vehicle capacity
    edge_capacity_mask = distance_matrix < total_capacity

    # 3. Clear Edge Evaluation
    # Define edge evaluation method based on distance and cumulative demand
    edge_evaluation = distance_matrix - cumulative_demand

    # Prioritize edges with positive evaluation values
    positive_evaluation_mask = edge_evaluation > 0

    # Combine all criteria into a single heuristic
    heuristic_matrix = edge_capacity_mask * positive_evaluation_mask * edge_evaluation

    return heuristic_matrix