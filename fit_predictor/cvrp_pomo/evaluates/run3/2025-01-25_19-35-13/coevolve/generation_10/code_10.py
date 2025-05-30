import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity
    
    # Create a distance matrix with the depot's distance to each customer
    with(torch.no_grad()):
        distance_matrix_with_depot = distance_matrix.clone()
        # The distance from the depot to itself should be zero, replace them with a large value
        distance_matrix_with_depot.fill_(float('inf'))
        distance_matrix_with_depot.scatter_(0, torch.tensor([0], dtype=torch.long), distance_matrix[0])

    # Calculate a basic heuristic based on normalized demands
    demand_heuristic = -normalized_demands

    # Calculate a potential function that encourages short distances
    potential_function = torch.exp(-distance_matrix_with_depot / distance_matrix_with_depot.mean())

    # Combine both heuristics with some weight
    combined_heuristic = demand_heuristic * potential_function

    # Cap the values at a certain threshold to avoid too large heuristics
    combined_heuristic = torch.clamp(combined_heuristic, min=-1, max=1)

    return combined_heuristic