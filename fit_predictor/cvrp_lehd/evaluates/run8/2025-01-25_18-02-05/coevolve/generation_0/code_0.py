import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Assuming that the depot is at index 0, and demands are normalized by the total vehicle capacity
    
    # Calculate the sum of the demands to normalize them by the total vehicle capacity
    total_demand = demands.sum()
    
    # Calculate the relative demand for each customer
    relative_demands = demands / total_demand
    
    # Calculate the distance from the depot to each customer and back (returning to the depot)
    return_distances = distance_matrix[:, 1:] + distance_matrix[1:, :]
    
    # Calculate the total distance if we visit each customer and return to the depot
    total_return_distances = return_distances.sum(dim=1)
    
    # Calculate the potential benefit for each edge (negative of the total distance to return to the depot)
    potential_benefit = -total_return_distances
    
    # Multiply the potential benefit by the relative demand to prioritize edges with high demand
    weighted_benefit = potential_benefit * relative_demands
    
    # Add a small constant to avoid division by zero or numerical instability
    epsilon = 1e-8
    weighted_benefit = torch.clamp(weighted_benefit, min=-epsilon)
    
    # Normalize the weighted benefit by the sum of all weighted benefits to ensure that the output
    # has the same shape as the distance matrix and that the sum of all elements is 1
    normalized_benefit = weighted_benefit / weighted_benefit.sum()
    
    return normalized_benefit