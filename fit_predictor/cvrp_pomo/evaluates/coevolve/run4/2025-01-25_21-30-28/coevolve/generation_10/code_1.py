import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Ensure that the distance matrix and demands are tensors
    distance_matrix = torch.tensor(distance_matrix, dtype=torch.float32)
    demands = torch.tensor(demands, dtype=torch.float32)

    # Ensure that the demands are normalized by the total vehicle capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity

    # Calculate the inverse of the distance matrix
    inv_distance_matrix = 1 / (distance_matrix + 1e-8)  # Adding a small constant to avoid division by zero

    # Compute the heuristics using a combination of Normalization and Inverse Distance heuristics
    # Normalization heuristic: Demands as weights
    normalization_heuristic = normalized_demands.unsqueeze(0).expand_as(distance_matrix)

    # Inverse Distance heuristic: Inverse of distances
    inverse_distance_heuristic = inv_distance_matrix

    # Combine the heuristics
    combined_heuristic = normalization_heuristic + inverse_distance_heuristic

    # Negative values for undesirable edges and positive values for promising ones
    heuristics = combined_heuristic - combined_heuristic.min()  # Shift the min to ensure all values are positive

    return heuristics