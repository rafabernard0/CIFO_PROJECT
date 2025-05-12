import random
import numpy as np
from copy import deepcopy

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


def custom_pmxo(parent1_repr, parent2_repr, start_window=2, end_window=5):
    """
    Perform custom partially mapped crossover between parent 1 and parent 2. Adapts PMXO to matrix.
    - Swaps columns [start_window:end_window] between the parents.
    - Resolves conflicts in columns [0, 1, 5, 6] by replacing duplicates.
    """

    children_1 = deepcopy(parent1_repr)
    children_2 = deepcopy(parent2_repr)

    # Swap the crossover window
    children_1[:, start_window:end_window] = parent2_repr[:, start_window:end_window]
    children_2[:, start_window:end_window] = parent1_repr[:, start_window:end_window]

    #Solve duplicates
    def fix_child(child, original_parent):
        fixed_child = child.deepcopy()
        forbidden_values = set(child[:, start_window:end_window].flatten())

        rows, cols = child.shape

        def find_non_conflicting_value(conflict_value, child_matrix, original_matrix):
            visited = set()
            current_value = conflict_value

            while True:
                if current_value in visited:
                    raise ValueError(f"Infinite loop detected for value {conflict_value}")
                visited.add(current_value)

                # Location of the value in the current child
                locations = np.argwhere(child_matrix == current_value)
                if len(locations) == 0:
                    return current_value

                row_idx, col_idx = locations[0]
                candidate = original_matrix[row_idx, col_idx]

                if candidate not in forbidden_values:
                    return candidate
                current_value = candidate

        for row in range(rows):
            for col in [0, 1, 5, 6]:
                val = fixed_child[row, col]
                if val in forbidden_values:
                    new_val = find_non_conflicting_value(val, child, original_parent)
                    fixed_child[row, col] = new_val
                    forbidden_values.add(new_val)

        return fixed_child

    child1_fixed = fix_child(children_1, parent1_repr)
    child2_fixed = fix_child(children_2, parent2_repr)

    return child1_fixed, child2_fixed