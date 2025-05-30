import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    
    # Normalize demand to get demand per unit of vehicle capacity
    normalized_demands = demands / demands.sum()
    
    # Normalize distances to make them comparable
    min_distance = distance_matrix.min()
    normalized_distances = (distance_matrix - min_distance) / (distance_matrix.max() - min_distance)
    
    # Calculate the initial heuristic score
    heuristic_scores = normalized_distances * normalized_demands
    
    # Adjust scores to have negative values for undesirable edges and positive values for promising ones
    heuristic_scores[distance_matrix == 0] = 0  # Set the heuristic score for the depot to zero
    heuristic_scores = -heuristic_scores
    
    return heuristic_scores