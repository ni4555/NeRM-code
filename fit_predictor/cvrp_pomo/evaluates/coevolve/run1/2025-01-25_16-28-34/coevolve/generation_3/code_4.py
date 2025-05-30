import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Demand and distance normalization
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity
    max_distance = distance_matrix.max()
    normalized_distances = distance_matrix / max_distance
    
    # Calculate the heuristic value
    heuristic_matrix = (normalized_distances * normalized_demands).subtract_(1)  # Subtracting 1 to make the matrix negative
    
    # Explicitly check vehicle capacities
    # Assuming that the distance matrix is symmetric and contains 0s on the diagonal (no distance to self)
    # We need to find a way to check vehicle capacities while using the heuristic matrix
    # A simple approach would be to sum the heuristic values for each row (vehicle's capacity)
    # and ensure that no row exceeds the total capacity. However, this is not efficient.
    # Instead, we can apply a threshold to the heuristic values based on the vehicle capacity.
    vehicle_capacity_threshold = 1.0  # This is a placeholder for the actual vehicle capacity
    capacity_normalized = normalized_demands / vehicle_capacity_threshold
    capacity_normalized = capacity_normalized.clamp(min=0)  # Ensure we don't have negative values
    
    # Combine the capacity normalization with the heuristic
    combined_heuristic_matrix = heuristic_matrix * capacity_normalized
    
    return combined_heuristic_matrix