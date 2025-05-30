import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize the demands to sum up to 1
    total_demand = demands.sum()
    normalized_demands = demands / total_demand

    # Calculate the heuristics based on the normalized demands
    # We use a simple heuristic: the demand of the customer multiplied by the distance
    # to the customer from the depot (which is 1 for all customers except the depot)
    # This heuristic is a simple greedy heuristic that prioritizes customers with higher demand
    # closer to the depot.
    heuristics = normalized_demands * distance_matrix

    # Subtract the maximum value from heuristics to ensure negative values for undesirable edges
    max_heuristic = heuristics.max()
    heuristics = heuristics - max_heuristic

    return heuristics