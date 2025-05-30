import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Ensure that demands are a column vector and expand to match the distance matrix shape
    demands = demands.view(-1, 1)
    
    # Calculate the heuristic values as the distance between the depot (0,0) and each customer
    # plus the absolute difference in demands multiplied by the maximum demand
    max_demand = torch.max(demands).item()
    heuristic_matrix = distance_matrix + torch.abs(demands - demands[:, 0].expand_as(demands)) * max_demand
    
    # Normalize the heuristic matrix so that it contains negative values for undesirable edges
    # and positive values for promising ones
    min_value = heuristic_matrix.min().item()
    max_value = heuristic_matrix.max().item()
    normalized_heuristic_matrix = (heuristic_matrix - min_value) / (max_value - min_value)
    
    return normalized_heuristic_matrix

# Example usage:
# distance_matrix = torch.tensor([[0, 2, 5, 1], [2, 0, 4, 3], [5, 4, 0, 2], [1, 3, 2, 0]])
# demands = torch.tensor([1, 2, 1, 2])
# heuristics = heuristics_v2(distance_matrix, demands)
# print(heuristics)