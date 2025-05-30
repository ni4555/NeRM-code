import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize demands to be between 0 and 1 based on the total vehicle capacity
    normalized_demands = demands / demands.sum()
    
    # Use a simple heuristic where the heuristic value is the product of the distance and the normalized demand
    # For edges leading to nodes with higher demand, the heuristic value will be higher, indicating a less promising edge
    # Since we want negative values for undesirable edges and positive for promising ones, we'll invert the demand scaling
    heuristic_matrix = distance_matrix * (1 - normalized_demands)
    
    return heuristic_matrix

# Example usage:
# Assuming distance_matrix and demands are PyTorch tensors of the correct shape
# distance_matrix = torch.tensor([[0, 2, 1], [2, 0, 3], [1, 3, 0]], dtype=torch.float32)
# demands = torch.tensor([0.5, 0.2, 0.3], dtype=torch.float32)
# result = heuristics_v2(distance_matrix, demands)
# print(result)