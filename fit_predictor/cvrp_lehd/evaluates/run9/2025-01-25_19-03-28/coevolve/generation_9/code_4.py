import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize demands
    demand_sum = demands.sum()
    normalized_demands = demands / demand_sum

    # Calculate the negative weighted distance based on demand
    negative_weighted_distance = -distance_matrix * normalized_demands.unsqueeze(1) * normalized_demands.unsqueeze(0)

    # Use dynamic load balancing to adjust negative weighted distance
    # Sort the distances by absolute value (to simulate load balancing)
    sorted_indices = torch.argsort(torch.abs(negative_weighted_distance), dim=1)
    negative_weighted_distance = negative_weighted_distance.index_select(1, sorted_indices)

    # Proximity-based route planning to enhance the heuristic
    # Increase the negative weight of distances to more distant customers
    proximity_factor = 1 / (distance_matrix.sum(dim=0) + 1e-6)
    negative_weighted_distance *= proximity_factor.unsqueeze(0)

    return negative_weighted_distance