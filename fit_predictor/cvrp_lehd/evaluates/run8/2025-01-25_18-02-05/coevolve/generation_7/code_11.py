import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Ensure that the demands are normalized to sum to 1
    normalized_demands = demands / demands.sum()

    # Initialize the heuristic matrix with zeros
    heuristic_matrix = torch.zeros_like(distance_matrix)

    # For each customer demand, determine the quantile and assign a value
    for i in range(len(demands)):
        for j in range(len(demands)):
            if i != j:  # Exclude the depot node
                # Use the quantile as the heuristic value
                quantile_value = torch.quantile(normalized_demands[i], 0.5)
                # Assign the quantile value to the corresponding edge
                heuristic_matrix[i, j] = quantile_value

    return heuristic_matrix

# Example usage:
# distance_matrix = torch.tensor([[0, 1, 2], [1, 0, 3], [2, 3, 0]])
# demands = torch.tensor([0.2, 0.3, 0.5])
# heuristics_matrix = heuristics_v2(distance_matrix, demands)
# print(heuristics_matrix)