import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the normalized demand for each customer
    normalized_demands = demands / demands.sum()

    # Initialize the heuristics matrix with zeros
    heuristics_matrix = torch.zeros_like(distance_matrix)

    # Calculate the heuristic values using a normalized demand-based method
    # This is a simple heuristic based on the idea that closer customers with lower demands
    # are more likely to be included in a route, hence we use negative values for distance
    # and positive values for demand normalized by total capacity.
    heuristics_matrix.fill_(distance_matrix.mean())  # Base heuristic
    heuristics_matrix += normalized_demands  # Adjust based on normalized demand

    # Apply a normalization to ensure all values are within a certain range
    # We can use Min-Max normalization or any other suitable method
    min_val = heuristics_matrix.min()
    max_val = heuristics_matrix.max()
    heuristics_matrix = (heuristics_matrix - min_val) / (max_val - min_val)

    return heuristics_matrix

# Example usage:
# Assuming a torch.Tensor for distance_matrix and demands
distance_matrix = torch.tensor([
    [0, 5, 10, 15],
    [5, 0, 6, 20],
    [10, 6, 0, 9],
    [15, 20, 9, 0]
])
demands = torch.tensor([3, 2, 1, 1])

# Get the heuristics
heuristic_values = heuristics_v2(distance_matrix, demands)

print(heuristic_values)