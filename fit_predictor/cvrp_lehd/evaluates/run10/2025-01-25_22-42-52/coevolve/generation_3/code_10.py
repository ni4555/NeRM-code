import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize the demands vector by the total vehicle capacity
    total_capacity = demands.sum()
    demands = demands / total_capacity
    
    # Initialize a matrix with zeros for the heuristic values
    heuristic_values = torch.zeros_like(distance_matrix)
    
    # Calculate the heuristic values based on the distance and the demands
    # We can use a simple heuristic like:
    # heuristic_values[i][j] = -distance_matrix[i][j] * demands[i] * demands[j]
    # This heuristic assigns a negative value to edges which would increase the cost
    # The more distant the edge, the more negative the value, discouraging its selection.
    # The value is also influenced by the product of demands at both ends, which encourages
    # edges that have high demand nodes at both ends.
    heuristic_values = -distance_matrix * demands.unsqueeze(1) * demands.unsqueeze(0)
    
    # We can also add a small positive constant to avoid division by zero
    epsilon = 1e-6
    heuristic_values = torch.clamp(heuristic_values, min=epsilon)
    
    return heuristic_values

# Example usage:
# Assuming distance_matrix and demands are PyTorch tensors with appropriate shapes
# distance_matrix = torch.tensor([[0, 1, 2], [1, 0, 3], [2, 3, 0]], dtype=torch.float32)
# demands = torch.tensor([0.2, 0.3, 0.5], dtype=torch.float32)
# result = heuristics_v2(distance_matrix, demands)
# print(result)