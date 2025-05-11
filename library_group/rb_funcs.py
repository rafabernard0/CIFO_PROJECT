import random
import numpy as np
from copy import deepcopy


def two_phase_shuffle_mutation(representation, mut_prob):
    """
    This mutation operator starts from a random point in the matrix and decides,
    with 50% probability, whether to shuffle a row or a column. Then, with mut_prob,
    it applies the shuffle. The operation continues through the entire matrix,
    wrapping around if necessary.
    """
    new_repr = deepcopy(representation)
    num_rows, num_cols = new_repr.shape

    # Starting point
    start_row = random.randint(0, num_rows - 1)
    start_col = random.randint(0, num_cols - 1)

    # Traverse the matrix from the starting point
    for i in range(num_rows):
        for j in range(num_cols):
            row = (start_row + i) % num_rows
            col = (start_col + j) % num_cols

            if random.random() < 0.5:
                # 50% chance to work on a row
                if random.random() < mut_prob:
                    np.random.shuffle(new_repr[row])
            else:
                # 50% chance to work on a column
                if random.random() < mut_prob:
                    new_repr[:, col] = np.random.permutation(new_repr[:, col])

    return new_repr
