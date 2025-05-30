import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    m = weight.shape[1]
    heuristic = np.zeros((prize.shape[0],))
    
    for i in range(prize.shape[0]):
        total_prize = 0
        total_weight = 0
        for j in range(i, prize.shape[0]):
            for k in range(m):
                total_weight += weight[j][k]
            if total_weight > 1:
                break
            total_prize += prize[j]
        heuristic[i] = total_prize
    
    return heuristic
