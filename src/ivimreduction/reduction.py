import itertools


class LinearReduction:
    """ Class to calculate the linear reduction with respect fitting method """
    # Class Variables for inviduals objects
    def __init__(self, bvals, data, n_reduction, fitting_method):
        self.bvals = bvals
        self.data = data
        self.n_reduction = n_reduction
        self.fitting_method = fitting_method

    # Class Variables for all objects
    bvals_best = [] # (bvals) x n_reduction
    rrmse_best = [] # 1 x n_reduction

    # Methods
    def linearAlgorithmLoop(data, bvals, n_reduction, fitting_method):
        """ A method for loop linearly in the best b_vals selections
        """
        def linearListSelection(input):
            """Return the n-1 possible iterations of an given list"""
            return itertools.combinations(input,len(input)-1)
        
        best_bvals =  bvals
        
        for i in range(n_reduction): 
            bvalsList = linearListSelection(best_bvals)
            # rrmse storage
            
            # mean_rrmse storage
            
            for j in range(len(bvalsList)):

                for x in range(len(data.shape[0])):
                    for y in range(len(data.shape[1])):
                        for z in range(len(data.shape[2])):
                            # fit
                            # calcule rrmse
                            rrmse_calculator(data_array, bvals, model, estimated_params)
                
                # mean rrmse
                mean_rrmse(rrmse_list)
            # select best_bval
        
            best_bvals = []
        
        return 0
        

    def rrmse_calculator(data_array, bvals, model, estimated_params):
        """ A method to calculate the

        data_array - np.array with 1 x len(bvals)
        bvals - np.array
        model - function(bvals, estimated_params)
            (D, D*, f, bvals) -> [1xlen(bvals)]
        estimated_params - np.array with 1x3 (D, D*, f)
        """

        return 0


    def mean_rrmse(data, bvals, model, estimated_params):
        """ A method to calculate the

        data_array - np.array with 1 x len(bvals)
        bvals - np.array
        model - function(bvals, estimated_params)
            (D, D*, f, bvals) -> [1xlen(bvals)]
        estimated_params - np.array with 1x3 (D, D*, f)
        """

        return 0
