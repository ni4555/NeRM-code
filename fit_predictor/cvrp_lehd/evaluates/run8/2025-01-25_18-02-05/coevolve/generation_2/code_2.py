import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Ensure demands are normalized by dividing by the sum of all demands
    demands_sum = demands.sum()
    demands = demands / demands_sum

    # Calculate the heuristics
    # The idea is to use the inverse of distance as a heuristic for the edges
    # and adjust it by customer demand, where higher demand edges are more promising
    heuristic_matrix = -torch.pow(distance_matrix, 2) * demands

    return heuristic_matrix