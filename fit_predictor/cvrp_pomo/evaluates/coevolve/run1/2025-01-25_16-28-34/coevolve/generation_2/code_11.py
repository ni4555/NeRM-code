import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the load for each edge based on the demands of the customers
    load = distance_matrix * demands
    
    # Normalize the load by the total vehicle capacity
    # Assuming that the demands vector represents the normalized demands
    total_capacity = demands.sum()
    
    # Define the heuristics: we want to favor edges with lower load and shorter distances
    # We use a cost function that penalizes high load and high distance
    heuristics = -load - distance_matrix
    
    # Adjust the heuristics to be more positive for promising edges and more negative for undesirable ones
    # This step is necessary because we want to use a PSO approach which is driven by positive heuristics
    # We normalize the heuristics by the total capacity to ensure that the cost is properly scaled
    heuristics = heuristics / total_capacity
    
    return heuristics