import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize the demands by the total vehicle capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity

    # Calculate the path potential based on distance and demand
    # Here we use a simple heuristic where the potential is inversely proportional to the demand
    # and directly proportional to the distance (i.e., longer distances have lower potential)
    path_potential = 1 / (distance_matrix + normalized_demands)

    # Normalize the path potential for consistent scaling
    # Here we use a simple normalization by the max potential value
    max_potential = torch.max(path_potential)
    normalized_potential = path_potential / max_potential

    # Introduce a penalty for high path potential to avoid overloading vehicles
    # This can be adjusted based on the desired level of load balancing
    load_balance_penalty = normalized_potential * (1 - normalized_demands)
    adjusted_potential = normalized_potential - load_balance_penalty

    # Return the adjusted potential matrix
    return adjusted_potential