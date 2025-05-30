import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize demands by the total vehicle capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity

    # Calculate the inverse distance heuristic
    # We use a large positive value for the diagonal to avoid self-loops
    inv_distance = 1.0 / (distance_matrix + 1e-6)

    # Calculate the demand normalization heuristic
    demand_norm = normalized_demands * inv_distance

    # Combine the heuristics to get the final heuristic values
    heuristic_values = demand_norm - inv_distance

    return heuristic_values