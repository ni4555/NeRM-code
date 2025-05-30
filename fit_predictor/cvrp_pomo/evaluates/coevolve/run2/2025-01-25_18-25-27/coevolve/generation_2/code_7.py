import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the demand difference matrix
    demand_diff = demands[:, None] - demands[None, :]
    
    # Calculate the cumulative demand difference
    cum_demand_diff = torch.cumsum(demand_diff, dim=1)
    
    # Calculate the absolute cumulative demand difference
    abs_cum_demand_diff = torch.abs(cum_demand_diff)
    
    # Calculate the potential edge scores
    edge_scores = -abs_cum_demand_diff + distance_matrix
    
    # Normalize the scores to ensure a balance between demand and distance
    min_score, max_score = edge_scores.min(), edge_scores.max()
    normalized_scores = (edge_scores - min_score) / (max_score - min_score)
    
    return normalized_scores