###########################################################
#                                                         #
#                                                       
#                                                         #
###########################################################
from itertools import combinations
import numpy as np
from tqdm import tqdm
from math import factorial


from subsampling.subsampling import subsampling, bvals_selection_image
from fits.nnls_fit import nnlsfit
from stats.stats import rsquared, cooksdistance


# 1. Linear
# 2. Combinatory


###########################################################
#                   Linear Reduction                      #
###########################################################


def linear_reduction(n, bvals, data):
    """
    Params:
        - n the number of bvals reduction
    """
    bvals_possible_reduction = bvals.tolist()

    # Create Bar of progress
    MAXCOUNT = sum([*range(len(bvals)-n,len(bvals),1)])*np.prod(data.shape[0:2])
    bar = tqdm(total=MAXCOUNT, position=0)

    for i in range(n):
        # Create a new list of bvals
        #   from subsampling/subsampling.py
                
        bvals_possible_reduction =  bvals_possible_reduction[1::]
        n_reduced = len(bvals_possible_reduction) - 1

        bvals_subsampling = subsampling(n_reduced, bvals_possible_reduction)
    

        cookdistance_list = []
        rsqred_list = np.zeros(len(bvals_subsampling))
        
        # Iter in bvals_subsampling
        #   from fits/nnls_fit.py
        for j in range(len(bvals_subsampling)):
            # Insert b=0 to all bvals subsampling:
            bvals_subsampling[j].insert(0,0)
            # save Params in .npy for every iter
            data_subsampling = bvals_selection_image(data, bvals_subsampling[j], bvals)
            # print(data_subsampling[0,0,0,:])
            d = np.zeros(data.shape[0:2])
            d_star = np.zeros(data.shape[0:2])
            f = np.zeros(data.shape[0:2])
            for k in range(data.shape[0]):
                for l in range(data.shape[1]):
                    for m in range(data.shape[2]):
                        d_star[k,l,m], d[k,l,m], f[k,l,m] = nnlsfit(data_subsampling[k,l,m], np.array(bvals_subsampling[j]), region=None)
                        # Analyse Stats
                        #   from stats/stats.py
                        params = [d[k,l,m], d_star[k,l,m], f[k,l,m]]
                        #rsqred_list[j] += rsquared(np.array(bvals_subsampling[j]), data_subsampling[k,l,m], params)
                        #cookdistance_list.append(cooksdistance(3 , np.array(bvals_subsampling[j]), data_subsampling[k,l,m], params, j))
        
                        bar.update()    
            save_fit(bvals_subsampling[j], d, d_star, f)
        rsqred_list = rsqred_list.tolist()
        
        #print(rsqred_list)
        
        rsq_idx_max = rsqred_list.index(max(rsqred_list))
        cookdistance_idx_max = cookdistance_list.index(max(cookdistance_list))
        
        best_bval = bvals_subsampling[rsq_idx_max]
        
        #print(bvals_subsampling[rsq_idx_max])
        print(bvals_subsampling[cookdistance_idx_max])
        
        # Select the best bvals and replace
        bvals_possible_reduction = best_bval

    bar.close()
    
    return best_bval


###########################################################
#                   Combinatory Reduction                 #
###########################################################

def combinatory_reduction(n, bvals, data):
    """
    Params:
        - n the number of bvals reduction
    """
    bvals_possible_reduction = bvals.tolist()
    
    # Create a new list of bvals
    #   from subsampling/subsampling.py
    bvals_possible_reduction = bvals_possible_reduction[1::]
    n_reduced = len(bvals_possible_reduction) - n
     
    bvals_subsampling = subsampling(n_reduced, bvals_possible_reduction)
   
    rsqred_list = np.zeros(len(bvals_subsampling))
   
    # Create Bar of progress
    MAXCOUNT = int(factorial(len(bvals_possible_reduction))/
                   (factorial(len(bvals_possible_reduction)-n)*factorial(n)))
    bar = tqdm(total=MAXCOUNT*np.prod(data.shape[0:2]), position=0)
    
    # Iter in bvals_subsampling
    #   from fits/nnls_fit.py
    for j in range(len(bvals_subsampling)):
        # save Params in .npy for every iter
        bvals_subsampling[j].insert(0,0)
        data_subsampling = bvals_selection_image(data, bvals_subsampling[j], bvals)
        for k in range(data.shape[0]):
            for l in range(data.shape[1]):
                for m in range(data.shape[2]):        
                    d_star, d, f = nnlsfit(data_subsampling[k,l,m], np.array(bvals_subsampling[j]), region=None)
   
                    # Analyse Stats
                    #   from stats/stats.py
                    params = [d, d_star, f]
                    rsqred_list[j] += rsquared(np.array(bvals_subsampling[j]), data_subsampling[k,l,m], params)
                    bar.update()
    bar.close()

    rsqred_list = rsqred_list.tolist()

    #print(len(rsqred_list))
    idx_max = rsqred_list.index(max(rsqred_list))
    best_bval = bvals_subsampling[idx_max]
    
    # Select the best bvals and replace
    bvals_possible_reduction = best_bval
    #print(len(best_bval))

    return best_bval



###########################################################
#                       TESTS                             #
###########################################################
def simul_signal(params_mean, std, bvals):
    """
    """

    d_mean, pd_mean, f_mean = params_mean

    signal = (1-f_mean)*np.exp(-bvals*d_mean)+f_mean*np.exp(-bvals*pd_mean)

    noise = np.random.normal(1, std, len(signal))

    signal_with_noise = signal+(signal*noise)

    return signal_with_noise

def simul_signal3d(params_mean, std, shape, bvals):
    """
    """

    d_mean, pd_mean, f_mean = params_mean
    data = np.zeros((shape[0],shape[1],shape[2],shape[3]))
    signal = (1-f_mean)*np.exp(-bvals*d_mean)+f_mean*np.exp(-bvals*pd_mean)
    
    for i in range(shape[0]):
            for j in range(shape[1]):
                    for k in range(shape[2]):
                        noise = np.random.normal(1, std, len(signal))
                        data[i,j,k,:] = signal+(signal*noise)

    return data


def linear_reduction_test():

    # define parameters of test
    params_mean = [0.001, 0.01, 0.2]
    bvals = np.array([0, 4, 8, 16, 30, 60, 120, 250, 500,
                      1000, 1200, 1400, 1600, 1800, 2000])
    snr = 0.1
    n = 5

    # Create a simulate signal
    sig = simul_signal(params_mean, snr, bvals)

    # run function
    best_bvals = linear_reduction(n, bvals, sig)

    print(best_bvals)

def combinatory_reduction_test():

    # define parameters of test
    params_mean = [0.001, 0.01, 0.2]
    bvals = np.array([0, 4, 8, 16, 30, 60, 120, 250, 500,
                      1000, 1200, 1400, 1600, 1800, 2000])
    snr = 0.1
    n = 5

    # Create a simulate signal
    sig = simul_signal(params_mean, snr, bvals)

    # run function
    best_bvals = combinatory_reduction(n, bvals, sig)

    print(best_bvals)
    
def linear_reduction_test3D():

    # define parameters of test
    params_mean = [0.001, 0.01, 0.2]
    bvals = np.array([0, 4, 8, 16, 30, 60, 120, 250, 500,
                      1000, 1200, 1400, 1600, 1800, 2000])
    snr = 0.1
    n = 5
    shape = [288,288,2,len(bvals)]
    
    # Create a simulate signal
    sig = simul_signal3d(params_mean, snr, shape, bvals)

    # run function
    best_bvals = linear_reduction(n, bvals, sig)

    print(best_bvals)

def combinatory_reduction_test3D():

    # define parameters of test
    params_mean = [0.001, 0.01, 0.2]
    bvals = np.array([0, 4, 8, 16, 30, 60, 120, 250, 500,
                      1000, 1200, 1400, 1600, 1800, 2000])
    snr = 0.1
    n = 5
    shape = [288,288,2,len(bvals)]
    
    # Create a simulate signal
    sig = simul_signal3d(params_mean, snr, shape, bvals)

    # run function
    best_bvals = combinatory_reduction(n, bvals, sig)

    print(best_bvals)
