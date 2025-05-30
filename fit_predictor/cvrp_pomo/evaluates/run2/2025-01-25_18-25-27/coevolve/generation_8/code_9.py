import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    total_capacity = demands.sum()

    # Normalize demands by total capacity
    normalized_demands = demands / total_capacity

    # Calculate the sum of demands along each edge, which is a measure of desirability
    edge_demand_sums = (normalized_demands[:, None] + normalized_demands[None, :]) * distance_matrix

    # Apply swap-insertion heuristic
    # Calculate the sum of demands for each pair of customers (including the depot)
    customer_demand_sums = edge_demand_sums.sum(dim=1)
    # Apply swap-insertion heuristic: prefer edges that would reduce the maximum demand
    swap_insertion heuristic = -torch.abs(customer_demand_sums[:, None] - customer_demand_sums[None, :])

    # Apply 2-opt heuristic
    # Calculate the sum of demands for each cycle
    cycle_demand_sums = edge_demand_sums.sum(dim=0)
    # Apply 2-opt heuristic: prefer edges that would reduce the maximum cycle demand
    two_opt_heuristic = -torch.abs(cycle_demand_sums[:, None] - cycle_demand_sums[None, :])

    # Combine heuristics with real-time penalties to prevent overloading
    # Introduce a penalty for each edge that would exceed the vehicle capacity
    over_capacity_penalty = (edge_demand_sums > 1).float() * 1000  # Example penalty

    # Combine all heuristics and penalties
    combined_heuristic = swap_insertion_heuristic + two_opt_heuristic + over_capacity_penalty

    return combined_heuristic