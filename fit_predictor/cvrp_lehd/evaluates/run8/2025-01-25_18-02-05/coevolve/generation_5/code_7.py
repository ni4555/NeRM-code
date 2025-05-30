import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the negative of the distance matrix to promote short paths
    negative_distance = -distance_matrix
    
    # Normalize the demands to sum to 1 (assuming total vehicle capacity is 1 for normalization)
    normalized_demands = demands / demands.sum()
    
    # Calculate the heuristics based on distance and demand
    # The idea is to penalize longer distances and balance the demands
    heuristics = negative_distance + torch.dot(negative_distance, normalized_demands)
    
    return heuristics