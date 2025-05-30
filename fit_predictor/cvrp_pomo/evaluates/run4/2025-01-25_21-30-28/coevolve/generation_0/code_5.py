import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the difference in demands between consecutive nodes
    diff_demands = demands[1:] - demands[:-1]
    
    # Calculate the cumulative sum of demands from the start to each node
    cum_demands = demands.cumsum(dim=0)
    
    # Calculate the cumulative sum of demands from the end to each node
    cum_demands_reverse = cum_demands[::-1].cumsum(dim=0)[::-1]
    
    # Calculate the potential reward for each edge based on the difference in demands
    # and the cumulative demands
    reward = diff_demands * (cum_demands - cum_demands_reverse)
    
    # Normalize the reward by the maximum reward to ensure all values are within a certain range
    max_reward = reward.max()
    normalized_reward = reward / max_reward
    
    # Subtract the normalized reward from 1 to get a negative value for undesirable edges
    # and a positive value for promising ones
    heuristics = 1 - normalized_reward
    
    return heuristics