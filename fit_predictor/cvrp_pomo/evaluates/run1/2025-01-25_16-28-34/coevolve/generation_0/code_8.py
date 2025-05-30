import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Initialize a matrix with the same shape as the distance matrix, filled with zeros
    heuristics = torch.zeros_like(distance_matrix)

    # Calculate the negative distance heuristic (more negative = better)
    heuristics = -distance_matrix

    # Normalize the heuristics based on demands (lower demand = more promising)
    # We use a scaling factor to prevent negative values from becoming too large
    demand_factor = demands.max()
    heuristics += (demands / demand_factor)

    # Subtract the normalized demand from the distance to penalize larger demands
    heuristics -= demands

    # Return the resulting heuristics matrix
    return heuristics