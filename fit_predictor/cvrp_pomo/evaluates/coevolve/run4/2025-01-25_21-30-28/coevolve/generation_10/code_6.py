import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    # Normalize the demands by the total vehicle capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity

    # Calculate the Normalization heuristic
    # Each customer has a demand that contributes positively to the heuristic
    normalization_heuristic = normalized_demands

    # Calculate the Inverse Distance heuristic
    # Each customer's distance to the depot is used to weigh its contribution negatively
    inverse_distance_heuristic = -distance_matrix[:, 0] / (distance_matrix[:, 0] ** 2 + 1e-8)

    # Combine the two heuristics using element-wise addition
    combined_heuristic = normalization_heuristic + inverse_distance_heuristic

    return combined_heuristic