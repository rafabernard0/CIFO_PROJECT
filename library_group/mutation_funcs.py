import random
import numpy as np
from copy import deepcopy

def block_rotation_mutation(representation, mut_prob):
    """
    Perform a mutation on the representation by rotating blocks of 4 artists.
    A 2x2 block is slided 35 times (from the left to the right and from up to down).
    """
    new_repr = deepcopy(representation)

    # There is 35 possibilities of occuring mutation (mutations are cumulative)
    for r in range(new_repr.shape[0]-1): # the block 2x2 slides down, after sliding all the way to the right
        for c in range(new_repr.shape[1]-1): # first, the block 2x2 slides to the right

            if random.random() <= mut_prob: # mutation occurs
                if (c == new_repr.shape[1]-2) or (r == new_repr.shape[0]-2):
                    # the last block sliding (down or to the right) will wrap around the matrix (completing the 2x2 block)
                    new_repr = np.append(new_repr, [new_repr[0]], axis=0) # add row
                    new_repr = np.append(new_repr, new_repr[:,0].reshape(-1, 1), axis=1) # add column
                    # perform mutation (rotation of the block)
                    new_repr[r:r+2, c:c+2] = np.rot90(new_repr[r:r+2, c:c+2]) #rotate 90 degrees clockwise
                    # substitute the first row and column by the last row and column, respectively (product of the wrapping)
                    new_repr[0,:] = new_repr[-1,:]
                    new_repr[:, 0] = new_repr[:, -1]
                    # remove the last row and column, that were added to allow the wrapping
                    new_repr = np.delete(new_repr, -1, axis=0)
                    new_repr = np.delete(new_repr, -1, axis=1)
                else:
                    new_repr[r:r+2, c:c+2] = np.rot90(new_repr[r:r+2, c:c+2]) #rotate 90 degrees clockwise
                    
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
