import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize the demands by the total vehicle capacity
    total_capacity = torch.sum(demands)
    normalized_demands = demands / total_capacity

    # Calculate the inverse distance heuristic (IDH) values
    idh_values = 1.0 / distance_matrix

    # Calculate the demand penalty function
    # Increase the cost for edges leading to vehicles that are near their capacity limits
    # We use a simple linear penalty here, but this can be adjusted as needed
    capacity_penalty_threshold = 0.8  # Vehicles are considered near capacity at 80% of capacity
    demand_penalty = normalized_demands * (1.0 - capacity_penalty_threshold)
    demand_penalty = torch.where(demand_penalty < 0, torch.zeros_like(demand_penalty), demand_penalty)
    demand_penalty = demand_penalty * torch.max(idh_values)

    # Combine IDH and demand penalty to get the heuristic values
    heuristic_values = idh_values - demand_penalty

    return heuristic_values