import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total demand
    total_demand = demands.sum()

    # Normalize the demands
    normalized_demands = demands / total_demand

    # Set the weight for balancing distance and demand
    alpha = 1.0  # This value can be tuned to emphasize one over the other

    # Calculate the heuristic value for each edge
    # We use a negative sign to represent costs, so a higher heuristic value is better
    # We use the negative of the distance because lower distances are preferable
    heuristic_matrix = -distance_matrix - alpha * normalized_demands.unsqueeze(1) * demands.unsqueeze(0)

    return heuristic_matrix