import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize demands by the total vehicle capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity

    # Inverse distance heuristic
    inverse_distance = 1 / distance_matrix

    # Load balancing heuristic
    load_balance = demands - normalized_demands

    # Combine heuristics with negative values for undesirable edges
    combined_heuristic = inverse_distance - load_balance

    # Ensure that the values are in the range of promising edges (positive values)
    combined_heuristic = torch.clamp(combined_heuristic, min=0)

    return combined_heuristic