import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total vehicle capacity
    total_capacity = demands.sum()
    
    # Normalize the demands to match the vehicle capacity
    normalized_demands = demands / total_capacity
    
    # Calculate the total demand of each edge
    edge_demands = (normalized_demands[:-1] * normalized_demands[1:]).sum(dim=1)
    
    # Calculate the fitness for each edge based on demand and distance
    # The fitness is negative because we want to maximize the negative value to indicate better edges
    fitness = -edge_demands
    
    # Add some heuristic to penalize long distances
    # Here we are using a simple linear function for demonstration purposes
    # The parameter 'distance_penalty' can be adjusted for different heuristics
    distance_penalty = 0.1
    fitness -= distance_penalty * distance_matrix
    
    # Ensure that the fitness for edges that do not contribute to the solution (self-edges) is set to a very low value
    # This is done by subtracting the maximum fitness from the fitness values of non-self-edges
    self_edge_fitness = fitness[self.is_nonzero(distance_matrix, dim=1)]
    non_self_edge_fitness = fitness[self.is_nonzero(distance_matrix, dim=0)]
    max_non_self_edge_fitness = non_self_edge_fitness.max()
    fitness[~self.is_nonzero(distance_matrix, dim=1)] -= max_non_self_edge_fitness
    
    return fitness

def is_nonzero(tensor, dim):
    # Vectorized way to check if the tensor is non-zero along a given dimension
    return tensor.ne(0).int().sum(dim=dim) > 0