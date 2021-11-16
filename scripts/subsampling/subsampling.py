###########################################################
#                                                         #
#                       Subsampling                       #
#                                                         #
###########################################################

# Import Modules
from itertools import chain, repeat, count, islice
from collections import Counter
import numpy as np


def combinations_without_repetition(r, iterable=None, values=None, counts=None):
    """ Generate combinations
    """
    if iterable:
        values, counts = zip(*Counter(iterable).items())
    f = lambda i, c: chain.from_iterable(map(repeat, i, c))
    n = len(counts)
    indices = list(islice(f(count(), counts), r))
    if len(indices) < r:
        return
    while True:
        yield list(values[i] for i in indices)
        for i, j in zip(reversed(range(r)), f(reversed(range(n)), reversed(counts))):
            if indices[i] != j:
                break
        else:
            return
        j = indices[i]+1
        for i, j in zip(range(i, r), f(count(j), counts[j:])):
            indices[i] = j


def subsampling(r, bvals):
    """
    Params:
        - b-value vector
        -

    Return:
        - subsampling of b-value vector
    """
    # Generate subsampling
    subsampling = list(combinations_without_repetition(r, iterable=bvals))
    
    return subsampling


def data_index(data, bvals_subsampling, bvals):
    """
    """
    
    if(len(data.shape)==1):
        data_new = np.zeros((len(bvals_subsampling)), dtype=object)
        bvals = bvals.tolist()
        for i in range(len(bvals_subsampling)):
            data_new[i] = bvals.index(bvals_subsampling[i])
    else:
        if(len(np.array(bvals_subsampling).shape) == 1):
            data_new = np.zeros((data.shape[0], data.shape[1], data.shape[2], len(bvals_subsampling)), dtype=object) 
            for i in range(len(bvals_subsampling)):
                #bvals = bvals.tolist()
                data_new[:, :, :, i] = data[:, :, :, bvals.tolist().index(bvals_subsampling[i])]    
        else: 
            data_new = np.zeros((data.shape[0], data.shape[1], data.shape[2], len(bvals_subsampling)), dtype=object) 
            for i in range(len(bvals_subsampling)):
                data_new[:, :, :, i] = data[:, :, :, bvals.index(bvals_subsampling[i])]

    return data_new


def bvals_selection_image(data, bvals_subsampling, bvals):
    """
    Params:
        - Data np.array [x,y,z,bvals]

    Return:
        - Data_new = np.array[x,y,z,bvals_subsampling]
    """
    data_subsampling = data_index(data, bvals_subsampling, bvals)

    return data_subsampling


###########################################################
#                                                         #
#                       Tests                             #
#                                                         #
###########################################################

def test_subsampling():
    """ Testing subsampling function
    """

    bvals = [0, 1, 2, 3, 4, 5]
    r = 5
    bvals_subsampling = [[0, 1, 2, 3, 4],
                         [0, 1, 2, 3, 5],
                         [0, 1, 2, 4, 5],
                         [0, 1, 3, 4, 5],
                         [0, 2, 3, 4, 5],
                         [1, 2, 3, 4, 5]]
    assert subsampling(r, bvals) == bvals_subsampling, "Error subsampling Cobinatory"
    print("subsampling function test passed")

def test_bvals_selection_image():
    """
    """
    
    bvals = [0, 1, 2, 3, 4, 5]
    r = 5
    bvals_subsampling = subsampling(r, bvals)
    data = np.zeros((288, 288, 10, 6), dtype=object)
    for i in range(len(bvals)):
        data[:, :, :, i] = bvals[i]
    data_new = bvals_selection_image(data, bvals_subsampling[0], bvals)
    assert data_new[0, 0, 0, :].tolist() == [0, 1, 2, 3, 4], "Error bvals_selection_image function"
    print("bvals_selection_image function test passed")
