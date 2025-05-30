import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Ensure the distance matrix and demands are on the same device
    if distance_matrix.device != demands.device:
        distance_matrix = distance_matrix.to(demands.device)
    
    # Calculate the total capacity of all vehicles
    total_capacity = demands.sum()
    
    # Normalize the demands to the capacity of each vehicle
    normalized_demands = demands / total_capacity
    
    # Calculate the load of each edge (demand * distance)
    load = normalized_demands.unsqueeze(1) * distance_matrix.unsqueeze(0)
    
    # Create a mask for desirable edges (load > 0)
    desirable_edges = load > 0
    
    # Create a mask for undesirable edges (load <= 0)
    undesirable_edges = load <= 0
    
    # Assign negative values to undesirable edges
    heuristics = -torch.ones_like(distance_matrix)
    heuristics[undesirable_edges] = 0
    
    # Calculate the maximum load for desirable edges
    max_load = desirable_edges.sum(dim=1)
    
    # Normalize the load to create a heuristic value for each edge
    heuristics[desirable_edges] = (load[desirable_edges] / max_load[desirable_edges]).squeeze()
    
    return heuristics