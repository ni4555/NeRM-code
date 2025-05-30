import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize the demands vector to sum to 1
    total_demand = demands.sum()
    normalized_demands = demands / total_demand

    # Calculate the "promising" score for each edge by taking the product of the inverse of distance and the demand
    # This heuristic assumes that closer nodes with higher demand are more promising
    promising_scores = (1.0 / distance_matrix) * normalized_demands

    # Subtract the maximum score from each element to avoid very high values (used to handle division by small distance values)
    # and to normalize the scores into a range that's easier to work with in optimization algorithms
    max_score = promising_scores.max()
    normalized_scores = promising_scores - max_score

    # Add the demand directly to encourage the inclusion of edges with high demand
    heuristics = normalized_scores + demands

    return heuristics