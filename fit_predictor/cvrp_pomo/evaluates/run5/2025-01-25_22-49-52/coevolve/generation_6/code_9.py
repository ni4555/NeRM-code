import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Ensure the demands are normalized by the total vehicle capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity

    # Calculate the potential value for each edge based on demand relaxation
    potential_values = distance_matrix * normalized_demands

    # Node partitioning to identify heavily loaded edges
    # Assuming that the higher the demand, the more promising the edge is
    heavily_loaded_edges = (potential_values > 0).float() * (potential_values * (1 + demands))

    # Path decomposition to reduce the problem size
    # Here we use a simple threshold to decide which edges to consider
    threshold = 0.5
    promising_edges = heavily_loaded_edges > threshold

    # Create a mask of the original distance matrix with the same shape
    mask = torch.ones_like(distance_matrix)

    # Apply the mask to the distance matrix to get the potential values
    # This will replace the original distances with potential values for promising edges
    heuristics_matrix = torch.where(promising_edges, potential_values, distance_matrix)

    return heuristics_matrix