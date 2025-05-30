import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize the demands to sum to 1, assuming the total vehicle capacity is 1
    demand_sum = demands.sum()
    normalized_demands = demands / demand_sum

    # Initialize the heuristics matrix with zeros
    heuristics = torch.zeros_like(distance_matrix)

    # Calculate the cost of traveling to each customer
    travel_costs = distance_matrix * normalized_demands

    # Calculate the heuristic values based on the travel costs
    heuristics = -travel_costs

    # Apply a normalization technique to ensure the heuristics are within a certain range
    heuristics = torch.exp(heuristics) / torch.exp(heuristics).sum()

    return heuristics