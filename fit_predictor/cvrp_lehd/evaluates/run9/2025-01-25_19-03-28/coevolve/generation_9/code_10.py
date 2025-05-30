import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Ensure the demands are normalized
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity

    # Initialize the heuristic matrix with zeros
    heuristic_matrix = torch.zeros_like(distance_matrix)

    # Calculate the negative weighted distance based on normalized demand
    negative_weighted_distance = -distance_matrix * normalized_demands

    # Add a small constant to avoid division by zero
    epsilon = 1e-8
    normalized_negative_weighted_distance = (negative_weighted_distance + epsilon) / (negative_weighted_distance + epsilon).sum()

    # Proximity-based heuristic: edges closer to the depot are more promising
    # We can use the inverse of the distance as the heuristic value
    inverse_distance = 1 / (distance_matrix + epsilon)

    # Combine the heuristics
    combined_heuristic = normalized_negative_weighted_distance + inverse_distance

    # Apply dynamic load balancing by reducing the heuristic value for edges that would exceed capacity
    for i in range(1, len(demands)):
        for j in range(1, len(demands)):
            if demands[i] > epsilon:
                # Calculate the current load if we include this edge
                current_load = demands[i]
                for k in range(1, len(demands)):
                    if k != i and k != j:
                        current_load += distance_matrix[i, j] * normalized_demands[k]
                # If the load exceeds capacity, reduce the heuristic value
                if current_load > 1:
                    combined_heuristic[i, j] = min(combined_heuristic[i, j], 0)

    return combined_heuristic