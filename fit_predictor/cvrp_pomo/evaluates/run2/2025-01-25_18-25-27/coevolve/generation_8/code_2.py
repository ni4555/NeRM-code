import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity

    # Calculate the initial heuristic based on customer demand
    demand_heuristic = normalized_demands

    # Incorporate distance heuristic by subtracting the distance
    distance_heuristic = -distance_matrix

    # Combine demand and distance heuristics
    combined_heuristic = demand_heuristic + distance_heuristic

    # Real-time penalties to prevent overloading
    # Assuming a penalty factor for each customer
    penalty_factor = torch.clamp(distance_matrix / max(distance_matrix), min=0, max=1)
    penalty_heuristic = penalty_factor * combined_heuristic

    # Apply normalization to the heuristics
    # Assuming that we want to prevent very large heuristics
    max_heuristic = torch.max(penalty_heuristic)
    normalized_heuristics = penalty_heuristic / max_heuristic

    return normalized_heuristics