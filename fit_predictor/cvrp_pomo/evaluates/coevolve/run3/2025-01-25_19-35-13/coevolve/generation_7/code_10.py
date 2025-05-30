import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    vehicle_capacity = demands.sum()  # Assuming the total demand is the vehicle capacity

    # Normalize demands relative to vehicle capacity
    normalized_demands = demands / vehicle_capacity

    # Initialize the potential matrix
    potential_matrix = torch.zeros_like(distance_matrix)

    # Compute the potential for each edge
    # Potential = sum of normalized demands + distance + road quality factor
    # For simplicity, we use a fixed road quality factor (e.g., 1.0)
    road_quality_factor = 1.0
    potential_matrix = normalized_demands.unsqueeze(1) + normalized_demands.unsqueeze(0) + \
                       distance_matrix + road_quality_factor

    # Refine the potential function to prevent division by zero errors
    potential_matrix = torch.clamp(potential_matrix, min=-1e6)

    # The heuristics matrix is the negative of the potential matrix
    # Negative values for undesirable edges, positive for promising ones
    heuristics_matrix = -potential_matrix

    return heuristics_matrix