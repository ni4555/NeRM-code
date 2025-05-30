import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize the demands by the total capacity (assumed to be 1 for simplicity)
    demands /= demands.sum()

    # Compute a simple potential value for each edge
    # This is a placeholder heuristic; it should be replaced with a more complex one
    # that considers all the mentioned factors such as node partitioning, demand relaxation, etc.
    potential_value = distance_matrix - demands

    # Add a penalty for edges connecting to the depot
    depot_penalty = -1e6  # arbitrary large penalty for depot connections
    potential_value += depot_penalty * (distance_matrix == 0).float()

    return potential_value