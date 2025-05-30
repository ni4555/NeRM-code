import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize the demands to be between 0 and 1
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity
    
    # Initialize the heuristic matrix with a default value (e.g., 0)
    heuristic_matrix = torch.zeros_like(distance_matrix)
    
    # Calculate the heuristic values based on the normalized demands
    # For simplicity, we'll use a simple heuristic: demand * distance
    # This is just a placeholder and can be replaced with more complex heuristics
    heuristic_matrix = normalized_demands * distance_matrix
    
    # Adjust the heuristic matrix to ensure negative values for undesirable edges
    # and positive values for promising ones
    # This is a simple thresholding approach, but more sophisticated methods can be used
    threshold = 0.5  # This threshold can be adjusted
    heuristic_matrix = torch.where(heuristic_matrix > threshold, heuristic_matrix, -heuristic_matrix)
    
    return heuristic_matrix