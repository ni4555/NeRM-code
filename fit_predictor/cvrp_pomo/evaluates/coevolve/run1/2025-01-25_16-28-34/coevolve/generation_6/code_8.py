import random
import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    # Normalize demands by the total vehicle capacity (assuming demands are normalized)
    # For this example, we'll use the maximum demand as a proxy for the total capacity
    total_capacity = demands.max()
    normalized_demands = demands / total_capacity
    
    # Initialize the heuristic matrix with zeros
    heuristic_matrix = torch.zeros_like(distance_matrix)
    
    # Apply Problem-specific Local Search
    # For simplicity, we'll use a basic heuristic that penalizes long distances
    # and high customer demands. We'll use a linear combination of distance and demand
    # for the heuristic value.
    heuristic_matrix += distance_matrix * (1 + normalized_demands)
    
    # Apply Adaptive PSO Population Management
    # PSO-inspired heuristic: We'll add a random component to the heuristic to encourage exploration
    # The random component will be scaled by the inverse of the distance (to encourage shorter paths)
    random_component = torch.rand_like(distance_matrix) * (1 / distance_matrix)
    heuristic_matrix += random_component
    
    # Apply Dynamic Tabu Search with Adaptive Cost Function
    # Tabu-inspired heuristic: We'll add a bonus for paths that haven't been visited recently
    # We'll simulate this by penalizing paths that have a high correlation with previously visited paths
    # For simplicity, we'll use a dummy tabu list to represent this
    tabu_list = torch.zeros_like(distance_matrix)
    tabu_penalty = torch.abs(tabu_list - 1)
    heuristic_matrix -= tabu_penalty
    
    return heuristic_matrix