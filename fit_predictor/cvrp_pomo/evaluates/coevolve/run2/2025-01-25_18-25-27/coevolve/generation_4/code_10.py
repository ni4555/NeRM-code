import torch
import numpy as np
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    demands_sum = demands.sum()
    normalized_demands = demands / demands_sum
    normalized_distance_matrix = distance_matrix.clone() / distance_matrix.max()
    
    # Step 1: Apply a normalization technique to the distance matrix
    # Negative values for undesirable edges, positive for promising ones
    # Using the difference from the average distance as a heuristic
    average_distance = normalized_distance_matrix.mean()
    distance_heuristic = normalized_distance_matrix - average_distance
    
    # Step 2: Incorporate customer demand into the heuristic
    # More demand means more negative heuristic (undesirable edge)
    demand_heuristic = -normalized_demands
    
    # Combine the two heuristics
    combined_heuristic = distance_heuristic + demand_heuristic
    
    # Ensure the output has the same shape as the input distance matrix
    return combined_heuristic