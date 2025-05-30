import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Demand-based heuristic: A higher demand suggests a higher priority for this customer
    demand_heuristic = demands.to(distance_matrix.dtype) / demands.sum()
    
    # Distance-based heuristic: A negative value for distance to encourage closer customers
    distance_heuristic = -distance_matrix
    
    # A simple combination of demand and distance heuristics
    heuristic_values = demand_heuristic + distance_heuristic
    
    # Clip the values to ensure we have a clear separation between good and bad edges
    # This prevents all edges having the same positive value due to high demand
    heuristic_values = torch.clamp(heuristic_values, min=-1.0, max=1.0)
    
    return heuristic_values

# Example usage:
# Assuming `distance_matrix` and `demands` are PyTorch tensors with appropriate shapes
# distance_matrix = torch.tensor([[...]], dtype=torch.float32)
# demands = torch.tensor([...], dtype=torch.float32)
# heuristics = heuristics_v2(distance_matrix, demands)
# print(heuristics)