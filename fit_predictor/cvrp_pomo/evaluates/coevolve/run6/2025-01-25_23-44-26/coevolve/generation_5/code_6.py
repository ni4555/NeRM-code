import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize the distance matrix to have a range between 0 and 1
    distance_matrix = (distance_matrix - distance_matrix.min()) / (distance_matrix.max() - distance_matrix.min())
    
    # Normalize the demands to be between 0 and 1
    normalized_demands = demands / demands.sum()
    
    # Calculate the inverse distance heuristic
    inverse_distance = 1 / distance_matrix
    
    # Apply a demand penalty function to high-demand customers
    demand_penalty = demands * (1 - demands)  # A simple quadratic penalty function
    
    # Combine the heuristics
    combined_heuristic = inverse_distance - demand_penalty
    
    # Normalize the combined heuristic to have a range between 0 and 1
    combined_heuristic = (combined_heuristic - combined_heuristic.min()) / (combined_heuristic.max() - combined_heuristic.min())
    
    return combined_heuristic