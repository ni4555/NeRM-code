import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize demands to be between 0 and 1
    demands_normalized = demands / demands.sum()
    
    # Calculate the negative of the demands to use as a penalty for longer distances
    demand_penalty = -torch.abs(demands_normalized)
    
    # Use the distance matrix as the base for the heuristic values
    heuristic_values = distance_matrix.clone()
    
    # Incorporate the demand penalty into the heuristic values
    heuristic_values += demand_penalty
    
    # Apply a simple heuristic to penalize edges that are longer than a certain threshold
    threshold = 0.5  # This is a hyperparameter that may need tuning
    longer_distance_penalty = torch.where(distance_matrix > threshold,
                                          torch.ones_like(distance_matrix) * 1000,  # Large penalty for longer distances
                                          torch.zeros_like(distance_matrix))
    heuristic_values += longer_distance_penalty
    
    # Return the resulting heuristic values
    return heuristic_values