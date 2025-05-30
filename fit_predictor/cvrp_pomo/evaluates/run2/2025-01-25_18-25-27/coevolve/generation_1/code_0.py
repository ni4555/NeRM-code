import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize the demands by the total vehicle capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity

    # Calculate the negative demand as a heuristic value for the edge
    negative_demand = -normalized_demands

    # Compute the distance heuristic
    distance_heuristic = distance_matrix

    # Combine the negative demand and distance heuristic
    combined_heuristic = negative_demand + distance_heuristic

    return combined_heuristic