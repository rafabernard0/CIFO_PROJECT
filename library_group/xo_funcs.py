import random
import numpy as np

def cyclic_crossover(parent1_repr, parent2_repr):
    """
    Cyclic Crossover
    """
    initial_random_idx = [random.randint(0, parent1_repr.shape[0] - 1), random.randint(0, parent1_repr.shape[1] - 1)]
    cycle_idxs = [initial_random_idx]
    current_cycle_idx = initial_random_idx
    while True:
        value_parent2 = parent2_repr[current_cycle_idx[0], current_cycle_idx[1]]
        next_cycle_idx = [np.where(parent1_repr==value_parent2)[0][0], np.where(parent1_repr==value_parent2)[1][0]]
        if next_cycle_idx in cycle_idxs:
            break
        cycle_idxs.append(next_cycle_idx)
        current_cycle_idx = next_cycle_idx

    offspring1_repr = np.zeros((parent1_repr.shape[0], parent2_repr.shape[1]))
    offspring2_repr = np.zeros((parent1_repr.shape[0], parent2_repr.shape[1]))

    for row in range(parent1_repr.shape[0]):
        for col in range(parent1_repr.shape[1]):
            if [row, col] in cycle_idxs:
                offspring1_repr[row, col] = parent1_repr[row, col]
                offspring2_repr[row, col] = parent2_repr[row, col]
            else:
                offspring1_repr[row, col] = parent2_repr[row, col]
                offspring2_repr[row, col] = parent1_repr[row, col]

    return offspring1_repr, offspring2_repr