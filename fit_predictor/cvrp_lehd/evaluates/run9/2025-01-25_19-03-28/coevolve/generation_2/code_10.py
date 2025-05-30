import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Assuming the demands have been normalized by the total vehicle capacity.
    
    # Calculate the total sum of the demand vector for normalization.
    total_demand = torch.sum(demands)
    
    # If the total demand is zero, we return a zero matrix as all edges are undesirable.
    if total_demand == 0:
        return torch.zeros_like(distance_matrix)
    
    # Compute the heuristic value for each edge.
    # We use the ratio of demand to distance for the heuristic. The idea is to prioritize
    # edges that have high demand relative to the distance between nodes. This is a simple
    # heuristic inspired by the Savings Algorithm, which can be adapted to include other
    # factors.
    heuristics = demands / distance_matrix
    
    # We can penalize longer distances by subtracting a value from the heuristic.
    # This value is chosen arbitrarily to give preference to shorter paths, but could be
    # adjusted based on problem specifics.
    distance_penalty = 1.0
    heuristics -= distance_penalty * distance_matrix
    
    # Normalize the heuristics to have a mean of 0 and a range between -1 and 1.
    # This is a common practice to ensure that the heuristics are within a useful range.
    min_val = torch.min(heuristics)
    max_val = torch.max(heuristics)
    heuristics = (heuristics - min_val) / (max_val - min_val)
    heuristics *= 2.0 - 1.0  # Scale between -1 and 1
    
    return heuristics