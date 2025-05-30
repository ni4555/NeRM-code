import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total demand to normalize
    total_demand = demands.sum()
    
    # Normalize the demands
    normalized_demands = demands / total_demand
    
    # Calculate the distance to the depot and to all other nodes
    distance_to_depot = distance_matrix[:, 0]  # Distance to the depot for all nodes
    distance_to_all = distance_matrix[:, 1:]  # Distance to all other nodes from the depot
    
    # Normalize the distances
    normalized_distances = distance_matrix / distance_to_depot[:, None]
    
    # Calculate the heuristic values based on the normalized demand and distance
    heuristic_values = normalized_demands * normalized_distances
    
    # Invert the heuristic to give higher weight to promising edges (negative values)
    heuristic_values = -heuristic_values
    
    # Apply a small constant to avoid division by zero in logarithmic functions
    epsilon = 1e-6
    heuristic_values = torch.clamp(heuristic_values, min=epsilon)
    
    return heuristic_values