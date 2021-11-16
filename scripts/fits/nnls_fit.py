###########################################################
#                                                         #
#               NNLS Fitting scripts                      #
#                                                         #
###########################################################

import numpy as np
from scipy.optimize import nnls
from scipy.signal import find_peaks
from tqdm import tqdm


def nnlsfit(data, bvals, region=None):
    """ Implement Non Negative Fitting for IVIM

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

    if(len(data.shape) == 1):
        # Alocate memory
        s = np.zeros(len(data))
        d_star = np.zeros(1)
        d = np.zeros(1)
        f = np.zeros(1)

        # prepare for NNLS
        NUMEL_D = 400
        DIFS = 10**(np.linspace(-3, 3, NUMEL_D))/1000
        M = np.zeros([len(bvals), 400])
        for i, bval in enumerate(bvals):
            for j, dif in enumerate(DIFS):
                M[i, j] = np.exp(-bval*dif)
        
        # NNLS Fitting
        if data[0] != 0:
            s = data[:]/data[0]
            f1, f2 = 0, 0
            sk = nnls(M, s)
            sk[0][0:25] = 0
            sk[0][375:400] = 0

            pks, _ = find_peaks(np.concatenate([[min(sk[0])], sk[0],  
                                                    [min(sk[0])]]))
            pks = pks-1

            for _, pk in enumerate(pks):

                sort = sorted(sk[0][pks], reverse=True)

                if sk[0][pk] == np.amax(sk[0][pks]):
                    d = DIFS[pk]
                    f1 = sk[0][pk]

                elif len(sort) >= 2 and sk[0][pk] == sort[1]:
                    d_star = DIFS[pk]
                    f2 = np.sum(sk[0][pk-25:pk+25])
            if (f1 + f2) == 0:
                f = 0
            else:
                f = f2/(f1 + f2)

        return d_star, d, f        
    else:
        # verify the region and set the bounds of iteration
        if region is None:
            x1, x2 = 0, data.shape[0]-1
            y1, y2 = 0, data.shape[1]-1
            z1, z2 = 0, data.shape[2]-1
        else:
            x1, x2 = region[0][0], region[0][1]
            y1, y2 = region[1][0], region[1][1]
            z1, z2 = region[2][0], region[2][1]

       # Alocate memory
        s = np.zeros(data.shape[3])
        d_star = np.zeros(data.shape[0:3])
        d = np.zeros(data.shape[0:3])
        f = np.zeros(data.shape[0:3])

        # prepare for NNLS
        NUMEL_D = 400
        DIFS = 10**(np.linspace(-3, 3, NUMEL_D))/1000
        M = np.zeros([len(bvals), 400])
        for i, bval in enumerate(bvals):
            for j, dif in enumerate(DIFS):
                M[i, j] = np.exp(-bval*dif)
        
        # NNLS Fitting
        for i in range(x1, x2):
            for j in range(y1, y2):
                for k in range(z1, z2):

                    # Fit
                    if data[i, j, k, 0] != 0:
                        s = data[i, j, k, :]/data[i, j, k, 0]
                        f1, f2 = 0, 0
                        sk = nnls(M, s)
                        sk[0][0:25] = 0
                        sk[0][375:400] = 0

                        pks, _ = find_peaks(np.concatenate([[min(sk[0])], sk[0],  
                                                            [min(sk[0])]]))
                        pks = pks-1

                        for _, pk in enumerate(pks):

                            sort = sorted(sk[0][pks], reverse=True)

                            if sk[0][pk] == np.amax(sk[0][pks]):
                                d[i, j, k] = DIFS[pk]
                                f1 = sk[0][pk]

                            elif len(sort) >= 2 and sk[0][pk] == sort[1]:
                                d_star[i, j, k] = DIFS[pk]
                                f2 = np.sum(sk[0][pk-25:pk+25])
                        if (f1 + f2) == 0:
                            f[i, j, k] = 0
                        else:
                            f[i, j, k] = f2/(f1 + f2)

        return d_star, d, f
