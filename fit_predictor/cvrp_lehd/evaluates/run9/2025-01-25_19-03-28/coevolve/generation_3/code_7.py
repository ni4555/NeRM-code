import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # The heuristic will be based on a combination of the distance and demand.
    # Negative values for undesirable edges and positive values for promising ones.
    # A simple heuristic could be to penalize long distances and high demands.
    
    # Calculate the distance penalty for each edge
    distance_penalty = 1 / (distance_matrix + 1e-8)  # Adding a small epsilon to avoid division by zero
    
    # Calculate the demand penalty for each customer
    demand_penalty = demands / (demands + 1e-8)  # Normalizing the demands to avoid division by zero
    
    # Combine the penalties: edges with longer distances and higher demands will have higher penalties
    heuristic_matrix = distance_penalty + demand_penalty
    
    return heuristic_matrix