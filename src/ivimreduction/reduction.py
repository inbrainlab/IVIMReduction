def hello():
    print("Hello World")


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
