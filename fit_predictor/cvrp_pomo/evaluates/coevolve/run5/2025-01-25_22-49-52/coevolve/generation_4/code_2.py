import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize distances by dividing by the maximum distance
    normalized_distance_matrix = distance_matrix / torch.max(distance_matrix)
    
    # Normalize demands by the total vehicle capacity
    total_capacity = demands[0]  # Assuming the first demand is the vehicle capacity
    normalized_demands = demands / total_capacity
    
    # Calculate potential values for explicit depot handling
    depot_potential = torch.sum(normalized_distance_matrix) / len(normalized_distance_matrix)
    
    # Calculate the potential value for each edge
    # This is a simple heuristic that assumes the potential value is a combination of distance and demand
    edge_potential_matrix = normalized_distance_matrix * normalized_demands
    
    # Add the depot potential to the edge potential matrix
    edge_potential_matrix += depot_potential
    
    # Integrate constraint programming and dynamic window approaches
    # This is a simplified approach where we subtract the total demand from the potential value
    # This will make the edges with high demand and high distance less desirable
    edge_potential_matrix -= torch.sum(normalized_demands)
    
    # Ensure the matrix has negative values for undesirable edges and positive values for promising ones
    # We can do this by setting the values below a certain threshold to -1 and the rest to 0
    threshold = 0.5
    edge_potential_matrix[edge_potential_matrix < threshold] = -1
    edge_potential_matrix[edge_potential_matrix >= threshold] = 0
    
    return edge_potential_matrix