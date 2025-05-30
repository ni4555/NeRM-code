import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize demands by the total vehicle capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity

    # Calculate the negative distance heuristic
    negative_distance_heuristic = -distance_matrix

    # Calculate the demand heuristic
    demand_heuristic = normalized_demands.unsqueeze(1) * distance_matrix.unsqueeze(0)

    # Combine the heuristics
    combined_heuristic = negative_distance_heuristic + demand_heuristic

    return combined_heuristic