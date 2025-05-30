import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Ensure the demands are normalized by the total vehicle capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity

    # Calculate the potential value for each edge
    # The potential value is a combination of distance and normalized demand
    # The heuristic is set to be positive for promising edges and negative for undesirable ones
    # Here we use a simple heuristic: negative distance and normalized demand to discourage long routes and high demands
    heuristics = -distance_matrix + normalized_demands

    # Normalize the heuristics to ensure they are within a certain range
    # This helps to avoid overflow or underflow in subsequent calculations
    heuristics = heuristics / heuristics.abs().max()

    return heuristics

# Example usage:
# Create a distance matrix and demands
distance_matrix = torch.tensor([[0, 2, 5, 1],
                                [2, 0, 3, 4],
                                [5, 3, 0, 2],
                                [1, 4, 2, 0]], dtype=torch.float32)
demands = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float32)

# Call the heuristics function
heuristic_values = heuristics_v2(distance_matrix, demands)

# Output the heuristic values
print(heuristic_values)