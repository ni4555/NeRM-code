import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Assuming the distance_matrix is symmetric and zero-diagonal
    num_nodes = distance_matrix.shape[0]
    heuristics = np.zeros_like(distance_matrix)
    
    # Compute the shortest path between each pair of nodes using Dijkstra's algorithm
    for i in range(num_nodes):
        # Use a priority queue to keep track of the nodes to visit
        visited = set()
        distances = {i: 0}
        priority_queue = [(0, i)]
        
        while priority_queue:
            current_distance, current_node = min(priority_queue)
            priority_queue.remove((current_distance, current_node))
            
            if current_node in visited:
                continue
            
            visited.add(current_node)
            
            for j in range(num_nodes):
                if j not in visited:
                    distance = current_distance + distance_matrix[current_node, j]
                    if j not in distances or distance < distances[j]:
                        distances[j] = distance
                        priority_queue.append((distance, j))
        
        # Calculate the heuristic values
        for j in range(num_nodes):
            heuristics[i, j] = distances[j] - distance_matrix[i, j]
    
    return heuristics