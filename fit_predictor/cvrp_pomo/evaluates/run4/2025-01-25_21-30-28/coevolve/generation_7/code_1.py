import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    # Calculate the total demand to normalize by
    total_demand = demands.sum()
    # Normalize demands
    normalized_demands = demands / total_demand
    # Inverse distance heuristic
    inverse_distance = 1 / distance_matrix
    # Demand normalization heuristic
    demand_heuristic = normalized_demands * demands
    # Combine heuristics
    combined_heuristic = inverse_distance + demand_heuristic
    # Clip values to ensure they are within a certain range (e.g., between -1 and 1)
    combined_heuristic = torch.clamp(combined_heuristic, min=-1, max=1)
    return combined_heuristic