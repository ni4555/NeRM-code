import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity

    # Calculate the potential value for each edge
    # We use a simple heuristic where the potential value is inversely proportional to the distance
    # and adjusted by the customer demand (normalized by total capacity)
    potential_value = (1 / (distance_matrix + 1e-8)) * normalized_demands.unsqueeze(1) * normalized_demands.unsqueeze(0)

    # We can add a penalty for edges that connect the depot to itself
    penalty = torch.eye(n)
    penalty[0, 0] = -1e9  # Penalize the depot-depot connection
    potential_value += penalty

    return potential_value