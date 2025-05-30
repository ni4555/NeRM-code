import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    # Calculate the demand contribution for each edge
    demand_contrib = torch.abs(demands[:, None] - demands[None, :])
    # Normalize demand contribution by the total vehicle capacity
    demand_contrib /= demands.sum()
    # Calculate the distance contribution for each edge
    distance_contrib = distance_matrix
    # Combine the demand and distance contributions
    combined_contrib = demand_contrib + distance_contrib
    # Normalize the combined contributions to have a range between -1 and 1
    min_contrib = combined_contrib.min()
    max_contrib = combined_contrib.max()
    normalized_contrib = 2 * (combined_contrib - min_contrib) / (max_contrib - min_contrib) - 1
    # Use an epsilon value to prevent division by zero
    epsilon = 1e-8
    # Avoid negative values for undesirable edges
    normalized_contrib = torch.clamp(normalized_contrib, min=epsilon)
    return normalized_contrib