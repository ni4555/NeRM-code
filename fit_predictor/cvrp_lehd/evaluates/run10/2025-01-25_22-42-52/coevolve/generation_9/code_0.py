import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Ensure the demand vector is normalized to the range [0, 1]
    demands_normalized = demands / demands.sum()
    
    # Calculate the potential reward for each edge
    # This is a simple heuristic that considers the difference in demand
    # between the nodes and the normalized demands
    edge_potential = distance_matrix * (demands - demands_normalized.unsqueeze(0))
    
    # Apply a threshold to convert the potential reward into a promising/undesirable indicator
    # This threshold can be adjusted depending on the specific case
    threshold = 0.1
    promising = edge_potential > threshold
    
    # Convert boolean mask to float tensor with positive values for promising edges
    # and negative values for undesirable edges
    # We subtract from 1 to convert True/False to 1/-1, then add 1 to shift the scale to [0, 1]
    heuristics_matrix = (promising.float() - 1) + 1
    
    return heuristics_matrix