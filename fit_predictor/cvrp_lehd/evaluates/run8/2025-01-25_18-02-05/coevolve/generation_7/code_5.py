import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the sum of the demands vector, which will be used to normalize the heuristics
    demand_sum = demands.sum()

    # Calculate the normalized demands
    normalized_demands = demands / demand_sum

    # Generate a matrix of ones with the same shape as the distance matrix
    ones_matrix = torch.ones_like(distance_matrix)

    # Calculate the heuristics by subtracting the normalized demands from the ones matrix
    heuristics = ones_matrix - normalized_demands

    # Ensure the heuristics have negative values for undesirable edges and positive ones for promising ones
    # The subtraction should yield negative values where the demand is high (edges to high demand customers)
    # and positive values where the demand is low (edges to low demand customers)
    
    # Convert any negative values to -1 and positive values to 1 for a binary heuristic representation
    heuristics = torch.clamp(heuristics, min=-1, max=1)

    return heuristics