import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the normalized demand
    normalized_demand = demands / demands.sum()
    
    # Generate a heuristics matrix based on normalized demand
    # Higher normalized demand corresponds to higher heuristics values
    heuristics_matrix = normalized_demand * (1 - distance_matrix)  # Negative values for longer distances
    
    return heuristics_matrix

# Example usage:
# distance_matrix = torch.tensor([[0, 10, 15, 20], [10, 0, 5, 10], [15, 5, 0, 5], [20, 10, 5, 0]], dtype=torch.float32)
# demands = torch.tensor([0.2, 0.4, 0.2, 0.2], dtype=torch.float32)
# heuristics = heuristics_v2(distance_matrix, demands)
# print(heuristics)