import random
import numpy as np
from copy import deepcopy


def cyclic_crossover(parent1_repr, parent2_repr):
    """
    Cyclic Crossover
    """

    # parent1_repr = parent1_repr.repr
    # parent2_repr = parent2_repr.repr

    initial_random_idx = [
        random.randint(0, parent1_repr.shape[0] - 1),
        random.randint(0, parent1_repr.shape[1] - 1),
    ]
    cycle_idxs = [initial_random_idx]
    current_cycle_idx = initial_random_idx
    while True:
        value_parent2 = parent2_repr[current_cycle_idx[0], current_cycle_idx[1]]
        next_cycle_idx = [
            np.where(parent1_repr == value_parent2)[0][0],
            np.where(parent1_repr == value_parent2)[1][0],
        ]
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


def custom_pmxo(parent1_repr, parent2_repr, verbose=False):
    """
    Perform custom partially mapped crossover between parent 1 and parent 2. Adapts PMXO to matrix.
    - Randomly chooses a crossover window size and position.
    - Swaps columns [start_window:end_window] between the parents.
    - Resolves conflicts in all columns outside the crossover window.
    """

    num_rows, num_cols = parent1_repr.shape

    # Crossover window definition
    window_size = random.randint(1, num_cols - 1)

    start_window = random.randint(0, num_cols - window_size)
    end_window = start_window + window_size

    if verbose:
        print(
            f"Custom Partially Mapped Crossover Window: from column {start_window} to column {end_window}"
        )

    children_1 = deepcopy(parent1_repr)
    children_2 = deepcopy(parent2_repr)

    # Swap the crossover window
    children_1[:, start_window:end_window] = parent2_repr[:, start_window:end_window]
    children_2[:, start_window:end_window] = parent1_repr[:, start_window:end_window]

    # Fix conflict in the children
    def fix_child(child, parent, start_window, end_window):
        fixed = deepcopy(child)
        num_rows, num_cols = fixed.shape
        crossover_cols = list(range(start_window, end_window))
        outside_cols = [col for col in range(num_cols) if col not in crossover_cols]

        for row in range(num_rows):
            used_values = set(
                fixed[row, crossover_cols]
            )  # values already in the crossover segment of that row

            for col in outside_cols:
                val = fixed[row, col]
                if val in used_values:
                    attempts = 0
                    max_attempts = num_cols * 2  # just a guard

                    while val in used_values and attempts < max_attempts:
                        rel_col = None
                        for c in crossover_cols:
                            if fixed[row, c] == val:
                                rel_col = c
                                break

                        if rel_col is None:
                            break  # should not happen

                        replacement = parent[row, rel_col]

                        if replacement not in fixed[row]:
                            val = replacement
                        else:
                            val = replacement  # keep looping if still duplicate

                        attempts += 1

                    fixed[row, col] = val
                    used_values.add(val)  # mark new value as used

        return fixed

    child1_fixed = fix_child(children_1, parent1_repr, start_window, end_window)
    child2_fixed = fix_child(children_2, parent2_repr, start_window, end_window)

    return child1_fixed, child2_fixed
