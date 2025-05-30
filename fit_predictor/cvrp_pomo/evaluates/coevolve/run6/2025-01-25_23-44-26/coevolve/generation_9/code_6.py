import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Ensure the demands are normalized by the total vehicle capacity
    total_capacity = demands.sum()
    if total_capacity == 0:
        raise ValueError("Total vehicle capacity must be non-zero.")
    normalized_demands = demands / total_capacity

    # Compute the inverse distance heuristic
    inverse_distance = 1 / (distance_matrix + 1e-6)  # Adding a small constant to avoid division by zero

    # Compute the demand-sensitive penalty (the penalty is inversely proportional to the normalized demand)
    demand_penalty = 1 / (normalized_demands + 1e-6)  # Adding a small constant to avoid division by zero

    # Combine the heuristics by taking a weighted sum
    # Weigh the demand penalty by a factor to balance it against the inverse distance heuristic
    demand_penalty_factor = 0.5  # This factor can be adjusted
    combined_heuristic = inverse_distance * demand_penalty_factor + demand_penalty * (1 - demand_penalty_factor)

    return combined_heuristic

# Example usage:
# distance_matrix = torch.tensor([[0, 1, 2, 3], [1, 0, 4, 5], [2, 4, 0, 6], [3, 5, 6, 0]], dtype=torch.float32)
# demands = torch.tensor([1, 2, 1, 2], dtype=torch.float32)
# heuristics = heuristics_v2(distance_matrix, demands)
# print(heuristics)