import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Initialize the heuristic matrix with zeros
    heuristic_matrix = np.zeros_like(distance_matrix)
    
    # Compute the shortest path between any two nodes using Dijkstra's algorithm
    for i in range(distance_matrix.shape[0]):
        # Initialize a priority queue with the first node
        priority_queue = [(0, i)]
        visited = set()
        
        while priority_queue:
            total_distance, node = heapq.heappop(priority_queue)
            
            if node not in visited:
                visited.add(node)
                
                for j in range(distance_matrix.shape[1]):
                    if j not in visited:
                        distance = distance_matrix[node, j]
                        if distance > 0:  # Exclude self-loops
                            new_distance = total_distance + distance
                            heapq.heappush(priority_queue, (new_distance, j))
                        
                # Update the heuristic matrix
                for j in range(distance_matrix.shape[1]):
                    if j not in visited:
                        heuristic_matrix[i, j] = distance_matrix[i, j] - total_distance
    
    return heuristic_matrix