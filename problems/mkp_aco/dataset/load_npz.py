import os

import numpy as np

if __name__ == '__main__':

    dataset_path = os.path.join('./', f"test100_dataset.npz")
    dataset = np.load(dataset_path)
    prizes, weights = dataset['prizes'], dataset['weights']
    print()
