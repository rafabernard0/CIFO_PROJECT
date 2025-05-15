import random
import numpy as np
from copy import deepcopy


def block_rotation_mutation(representation, mut_prob):
    """
    Perform a mutation on the representation by rotating 2x2 blocks.
    - The block slides across the matrix (left to right, top to bottom).
    - Each block has a `mut_prob` chance of being rotated 90° clockwise.
    - Edge blocks wrap around (toroidal grid).
    """
    new_repr = deepcopy(representation)
    rows, cols = new_repr.shape

    for r in range(rows):
        for c in range(cols):
            if random.random() <= mut_prob:
                # defining the 2x2 block (with wrap-around for edges)
                block = np.array([
                    [new_repr[r, c], new_repr[r, (c + 1) % cols]],
                    [new_repr[(r + 1) % rows, c], new_repr[(r + 1) % rows, (c + 1) % cols]]
                ])
                
                # rotating the block 90° anti-clockwise
                rotated_block = np.rot90(block)
                
                # placing the rotated block back (with wrap-around)
                new_repr[r, c] = rotated_block[0, 0]
                new_repr[r, (c + 1) % cols] = rotated_block[0, 1]
                new_repr[(r + 1) % rows, c] = rotated_block[1, 0]
                new_repr[(r + 1) % rows, (c + 1) % cols] = rotated_block[1, 1]
    
    return new_repr


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


def semi_shuffle(representation, mut_prob, verbose=False):
    """
    Semi-shuffle mutation: Randomly select a number of slots in each stage and shuffle them.
    This mutaation is applied with a probability of mut_prob.
    """
    if random.random() > mut_prob:
        return deepcopy(representation)

    new_repr = deepcopy(representation)

    n_stages, n_slots = new_repr.shape

    # 1) Choose block size
    block_size = random.randint(1, n_slots - 1)

    # 2) For each row, pick a contiguous block [start, start+block_size)
    idxs_per_stage = []
    for _ in range(n_stages):
        start = random.randint(0, n_slots - block_size)
        idxs_per_stage.append(list(range(start, start + block_size)))

    # 3) Build a random permutation of the rows
    perm = list(range(n_stages))
    random.shuffle(perm)

    # 4) Move each row’s block into the other row’s block positions
    for src_row in range(n_stages):
        dst_row = perm[src_row]
        src_cols = idxs_per_stage[src_row]
        dst_cols = idxs_per_stage[dst_row]
        # copy the values from src_row/src_cols into new_repr at dst_row/dst_cols
        new_repr[dst_row, dst_cols] = representation[src_row, src_cols]

    if verbose:
        print("Block Size:")
        print(block_size)
        print("Indices per Stage:")
        print(idxs_per_stage)
        print("Permutation:")
        print(perm)
        print("Original Representation:")
        print(representation)
        print("New Representation after Semi-Shuffle:")
        print(new_repr)

    return new_repr
