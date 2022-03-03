import numpy as np
from .nnls_fit import nnlsfit

from scipy.optimize import curve_fit
from scipy.signal import find_peaks

from tqdm import tqdm


def nonlinearfit(data, bvals, region=None):
    """ Implement Non Linear Fitting for IVIM
        Parameters
        ----------
	        data: array-like, 4D Matrix
            bvals: array-like
            region: array-like, [(x1,x2),(y1,y2),(z1,z2)]
            optional (default= None)
        Returns
        -------
            d_star: 3D Matrix
                The Pseudo Diffusion Coefficient.
            d: 3D Matrix
                The Diffusion Coefficient.
            pfraction: 3D Matrix
                The Perfusion Fraction.
        References
        ----------
        [1] -
    """

    # find the initial conditions with nnls fit
    d_star0, d0, pfraction0 = nnlsfit(data, bvals, region)

    # verify the region and set the bounds of iteration
    if region is None:
        x1, x2 = 0, data.shape[0]-1
        y1, y2 = 0, data.shape[1]-1
        z1, z2 = 0, data.shape[2]-1
    else:
        x1, x2 = region[0][0], region[0][1]
        y1, y2 = region[1][0], region[1][1]
        z1, z2 = region[2][0], region[2][1]

    # Create Bar of progress and alocate memory
    MAXCOUNT = (x2-x1)*(y2-y1)*(z2-z1)    
    bar = tqdm(total=MAXCOUNT, position=0)

    s = np.zeros(data.shape[3])
    d_star = np.zeros(data.shape[0:3])
    d = np.zeros(data.shape[0:3])
    pfraction = np.zeros(data.shape[0:3])

    # Non Linear Least Squares Fitting
    for i in range(x1, x2):

        for j in range(y1, y2):

            for k in range(z1, z2):

                if data[i, j, k, 0] != 0:
                    while True:
                        s = data[i, j, k, :] / data[i, j, k, 0]
                        p0 = [d0[i, j, k], d_star0[i, j, k], pfraction0[i, j, k]]
                        try:
                            popt = curve_fit(func, bvals, s, p0=p0)
                        except RuntimeError:
                            while True:
                                p0 = [0.001,0.015,0.2]
                                try:
                                    popt = curve_fit(func, bvals, s, p0=p0)
                                except RuntimeError:
                                    popt = [ [d0[i, j, k], d_star0[i, j, k], pfraction0[i, j, k]] , [[0,0],[0,0]] ]
                                    break
                                break
                        break
                    d[i, j, k] = popt[0][0]
                    d_star[i, j, k] = popt[0][1]
                    pfraction[i, j, k] = popt[0][2]   
                bar.update()

    bar.close()

    return d_star, d, pfraction
