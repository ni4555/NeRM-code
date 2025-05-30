import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the savings for each edge
    savings = distance_matrix - (2 * demands.unsqueeze(1) * demands.unsqueeze(0))
    
    # Normalize savings by the maximum savings to get the ratio of savings to maximum savings
    max_savings = savings.max()
    normalized_savings = savings / max_savings
    
    # Apply a penalty to edges with negative savings
    penalty_threshold = 0.1
    penalty = torch.where(savings < -penalty_threshold, -max_savings, 0)
    penalized_savings = normalized_savings + penalty
    
    # Apply a discount to the normalized savings to encourage early deliveries
    discount_factor = 0.5
    discounted_savings = penalized_savings * discount_factor
    
    return discounted_savings