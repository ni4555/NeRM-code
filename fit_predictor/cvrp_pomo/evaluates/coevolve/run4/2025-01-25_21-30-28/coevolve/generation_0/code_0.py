import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Ensure the distance_matrix and demands are tensors
    distance_matrix = torch.tensor(distance_matrix, dtype=torch.float32)
    demands = torch.tensor(demands, dtype=torch.float32)

    # Calculate the total vehicle capacity (sum of demands)
    total_capacity = demands.sum()

    # Calculate the heuristic values for each edge
    # Inverse of distance to encourage short paths
    # Normalize by total_capacity to account for vehicle capacity
    # Subtract the demand to make high demand nodes less promising
    heuristic_values = (1 / (distance_matrix + 1e-6)) * (demands / total_capacity) - demands

    # Ensure the heuristic values are within the desired range (e.g., negative for undesirable edges)
    # For example, we can use the minimum negative value as the threshold for undesirable edges
    min_promising_value = torch.min(heuristic_values[heuristic_values > 0])
    heuristic_values[heuristic_values <= 0] = -min_promising_value
    heuristic_values[heuristic_values > 0] += min_promising_value

    return heuristic_values