import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Demand-driven heuristic: Promote edges leading to customers with high demand
    demand_heuristic = demands[1:] - demands[1:].mean()

    # Distance-driven heuristic: Discourage long edges
    distance_heuristic = -distance_matrix

    # Normalize both heuristics to the same scale (0 to 1)
    min_val = distance_heuristic.min()
    max_val = distance_heuristic.max()
    distance_heuristic = (distance_heuristic - min_val) / (max_val - min_val)
    
    # Combine heuristics: Here we could use any combination strategy. For simplicity, we take the minimum of the two.
    combined_heuristic = torch.clamp_min(distance_heuristic + demand_heuristic, min=0)
    
    # Adjust the heuristic values to ensure some edges are penalized (negative values)
    # For demonstration purposes, we can just subtract the minimum of combined heuristic from all values
    adjusted_heuristic = combined_heuristic - combined_heuristic.min()
    
    return adjusted_heuristic

# Example usage:
# n = number of customers (including the depot)
# Example distance matrix and demands vector:
# distance_matrix = torch.tensor([[0, 2, 5, 7], [2, 0, 3, 8], [5, 3, 0, 1], [7, 8, 1, 0]])
# demands = torch.tensor([0, 2, 1, 1])  # 0 for depot, then customer demands
# heuristics = heuristics_v2(distance_matrix, demands)
# print(heuristics)