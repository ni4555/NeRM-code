import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Step 1: Precise Demand Handling
    # Calculate cumulative demand mask
    total_demand = demands.sum()
    cumulative_demand = torch.cumsum(demands, dim=0)
    demand_mask = cumulative_demand / total_demand

    # Step 2: Capacity Constraint Prioritization
    # Calculate edge feasibility mask
    max_capacity = distance_matrix.sum() / 2  # Assuming two-way distance matrix
    edge_capacity_mask = (distance_matrix < max_capacity).float()

    # Step 3: Clear Edge Evaluation
    # Define edge evaluation based on distance and demand
    edge_evaluation = distance_matrix * demand_mask * edge_capacity_mask

    # Step 4: Optimization Strategies
    # Directly optimize by combining the above steps
    # Subtract the values to make larger negative values for undesirable edges
    optimized_heuristics = -edge_evaluation.sum(dim=1)

    return optimized_heuristics