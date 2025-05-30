import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize demands by the total vehicle capacity (assuming total capacity is 1 for simplicity)
    normalized_demands = demands / demands.sum()
    
    # Initialize a tensor of zeros with the same shape as distance_matrix
    heuristics = torch.zeros_like(distance_matrix)
    
    # Calculate the inverse distance heuristic (IDH) for each edge
    # We use a simple inverse of the distance as the heuristic value
    heuristics = 1 / (distance_matrix + 1e-8)  # Adding a small constant to avoid division by zero
    
    # Integrate demand normalization into the heuristic
    # We multiply the IDH by the normalized demand of the destination node
    heuristics *= normalized_demands
    
    # Implement a penalty function for edges that are close to the vehicle's capacity
    # We use a simple linear penalty proportional to the distance
    # This is a simplified example, and the actual penalty function can be more complex
    penalty_threshold = 0.8  # Threshold for when the penalty should be applied
    penalty_factor = 1.5  # Factor by which the heuristic is penalized
    heuristics[distance_matrix > penalty_threshold] *= penalty_factor
    
    return heuristics