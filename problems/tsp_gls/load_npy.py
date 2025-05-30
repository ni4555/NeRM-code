import os

import numpy as np


def load_dataset(fp):
    data = np.load(fp)
    return data


if __name__ == '__main__':
    mood = 'test'
    problem_size = '200'
    basepath = os.path.dirname(__file__)
    dataset_path = os.path.join(basepath, f"dataset/{mood}{problem_size}_dataset.npy")
    dataset = load_dataset(dataset_path)
    print(dataset)
