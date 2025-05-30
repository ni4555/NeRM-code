import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize demands by the sum of demands to prevent division by zero
    total_demand = demands.sum()
    normalized_demands = demands / total_demand

    # Calculate potential function weights based on normalized demands, distance, and road quality
    # Assuming distance_matrix contains distances and some road quality information
    # For simplicity, we will use distance as the road quality factor
    # The potential function is defined as: potential = normalized_demand * distance
    potential_weights = normalized_demands * distance_matrix

    # Introduce a penalty for large distances (assuming that edges with high distance are less desirable)
    # This is a simple heuristic to avoid overly long paths
    penalty_factor = 1.0  # Adjust this factor as needed
    potential_weights = potential_weights - penalty_factor * distance_matrix

    # Subtract the potential weights for edges that connect to the depot to ensure they are not favored
    depot_index = 0  # Index of the depot node
    potential_weights[:, depot_index] -= potential_weights[depot_index, :]

    return potential_weights