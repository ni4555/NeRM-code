import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize distance matrix by dividing by the maximum distance to ensure the values are in the range [0, 1]
    normalized_distance_matrix = distance_matrix / torch.max(distance_matrix)
    
    # Calculate the potential reward for each edge (distance * demand)
    potential_reward_matrix = normalized_distance_matrix * demands
    
    # Subtract the demand from the potential reward to penalize edges that would exceed vehicle capacity
    reward_matrix = potential_reward_matrix - demands
    
    # Apply a threshold to convert the reward matrix into a heuristics matrix
    # Edges with a positive value are considered promising, while edges with a negative value are undesirable
    threshold = 0.5
    heuristics_matrix = torch.where(reward_matrix > threshold, reward_matrix, -torch.inf)
    
    return heuristics_matrix