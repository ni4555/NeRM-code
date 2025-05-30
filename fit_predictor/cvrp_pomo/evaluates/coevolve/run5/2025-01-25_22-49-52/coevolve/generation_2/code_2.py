import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    # Initialize the heuristic matrix with zeros
    heuristic_matrix = torch.zeros_like(distance_matrix)
    
    # Node partitioning: Create a partitioning of nodes based on demand
    demand_threshold = 0.5  # Threshold for partitioning nodes
    partitioned_nodes = (demands > demand_threshold).nonzero(as_tuple=False)
    
    # Demand relaxation: Relax demands of nodes in partitioned_nodes
    relaxed_demands = demands.clone()
    relaxed_demands[partitioned_nodes] *= 0.9
    
    # Path decomposition: Decompose the problem into smaller subproblems
    subproblem_matrices = []
    for i in range(n):
        for j in range(i + 1, n):
            subproblem_matrix = distance_matrix[i, j] - relaxed_demands[i] * relaxed_demands[j]
            subproblem_matrices.append(subproblem_matrix)
    
    # Constraint programming: Apply constraint programming to each subproblem
    for subproblem_matrix in subproblem_matrices:
        # Assuming a simple constraint programming approach: minimize distance
        # Here we use a simple heuristic which is to take the minimum distance
        min_distance = torch.min(subproblem_matrix)
        heuristic_matrix[i, j] = min_distance
        heuristic_matrix[j, i] = min_distance
    
    # Dynamic window techniques: Adjust the heuristic matrix based on the current vehicle capacities
    # For simplicity, we'll just use the average distance to the nearest depot
    for i in range(1, n):
        nearest_depot_distance = torch.min(distance_matrix[i, :1])
        heuristic_matrix[i, 0] = nearest_depot_distance
        heuristic_matrix[0, i] = nearest_depot_distance
    
    # Multi-objective evolutionary algorithms: Optimize load balancing
    # This step is conceptual as it requires complex evolutionary algorithm implementation
    # Here we just assign a positive value to encourage load balancing
    for i in range(n):
        for j in range(i + 1, n):
            if (demands[i] + demands[j]) <= 1.0:
                heuristic_matrix[i, j] += 0.1
                heuristic_matrix[j, i] += 0.1
    
    return heuristic_matrix