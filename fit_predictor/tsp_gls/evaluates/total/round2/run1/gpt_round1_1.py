import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Calculate the degree of each node
    degrees = np.sum(distance_matrix, axis=0)
    
    # Calculate the betweenness centrality for each edge
    betweenness_centrality = np.zeros(distance_matrix.shape)
    for k in range(distance_matrix.shape[0]):
        for i in range(distance_matrix.shape[0]):
            for j in range(distance_matrix.shape[0]):
                if distance_matrix[i, j] != 0:
                    betweenness_centrality[i, j] += (distance_matrix[i, k] * distance_matrix[k, j]) / (np.sum(distance_matrix[k, :] != 0))
    
    # Normalize the betweenness centrality
    max_betweenness = np.max(betweenness_centrality)
    betweenness_centrality = betweenness_centrality / max_betweenness
    
    # Adjust the distance matrix based on the betweenness centrality
    adjusted_distance_matrix = distance_matrix.copy()
    for i in range(adjusted_distance_matrix.shape[0]):
        for j in range(adjusted_distance_matrix.shape[0]):
            adjusted_distance_matrix[i, j] *= (1 - betweenness_centrality[i, j])
    
    # Introduce mutation with a focus on nodes with higher degrees
    mutation_strength = 0.05
    high_degree_indices = np.argsort(degrees)[::-1][:int(mutation_strength * distance_matrix.shape[0])]
    mutation_indices = np.random.choice(high_degree_indices, size=int(mutation_strength * distance_matrix.shape[0] / 2), replace=False)
    for i in mutation_indices:
        j = np.random.randint(0, distance_matrix.shape[0])
        adjusted_distance_matrix[i, j] = np.random.rand()
    
    # Add a second mutation phase to avoid getting trapped in a local minimum
    for i in mutation_indices:
        j = np.random.randint(0, distance_matrix.shape[0])
        adjusted_distance_matrix[i, j] = np.random.uniform(low=0.5, high=1.5)
    
    return adjusted_distance_matrix
