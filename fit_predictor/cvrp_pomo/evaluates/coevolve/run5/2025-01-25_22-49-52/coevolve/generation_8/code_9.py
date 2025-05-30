import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Ensure demands are normalized
    total_capacity = demands.sum()
    if total_capacity == 0:
        raise ValueError("Total vehicle capacity cannot be zero.")
    
    normalized_demands = demands / total_capacity
    
    # Compute the heuristic values based on distance and demand
    # Here, we use a simple heuristic that penalizes edges with higher distances
    # and higher demands, favoring shorter and less loaded edges.
    # This is just an example heuristic and may not be suitable for all DCVRP instances.
    
    # Initialize the heuristic matrix with high negative values for undesirable edges
    heuristic_matrix = torch.full(distance_matrix.shape, fill_value=-1e6)
    
    # Update the heuristic matrix for each edge based on distance and demand
    # We use a linear combination of distance and demand to calculate the heuristic
    # The coefficients are set arbitrarily for demonstration purposes
    distance_weight = 0.5
    demand_weight = 0.5
    
    for i in range(distance_matrix.size(0)):
        for j in range(distance_matrix.size(1)):
            if i != j and i != 0:  # Exclude the depot node from the heuristic
                # Calculate the heuristic value for the edge (i, j)
                edge_heuristic = distance_matrix[i][j] * distance_weight + \
                                 normalized_demands[j] * demand_weight
                # Set the heuristic value for the edge (i, j)
                heuristic_matrix[i][j] = edge_heuristic
    
    return heuristic_matrix

# Example usage:
# distance_matrix = torch.tensor([[0, 1, 2, 3], [1, 0, 4, 5], [2, 4, 0, 6], [3, 5, 6, 0]])
# demands = torch.tensor([0.2, 0.4, 0.3, 0.1])
# print(heuristics_v2(distance_matrix, demands))