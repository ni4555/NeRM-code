import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Ensure that the distance matrix and demands are both tensors
    distance_matrix = torch.tensor(distance_matrix, dtype=torch.float32)
    demands = torch.tensor(demands, dtype=torch.float32)
    
    # Calculate the total demand normalized by the vehicle capacity
    total_demand = demands.sum()
    
    # Calculate the heuristics based on the distance and demands
    # Here, we use a simple heuristic that encourages edges with lower distance and lower demand
    heuristics = (1 / (distance_matrix + 1e-6)) * (1 - demands / total_demand)
    
    return heuristics

# Example usage:
# distance_matrix = torch.tensor([[0, 1, 2, 3], [1, 0, 4, 5], [2, 4, 0, 6], [3, 5, 6, 0]])
# demands = torch.tensor([1, 2, 1, 1])
# print(heuristics_v2(distance_matrix, demands))