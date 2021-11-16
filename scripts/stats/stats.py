###########################################################
#                                                         #
#               Statistics Implementations                #
#                                                         #
###########################################################

import numpy as np
from fits.nonlinfit import nonlinearfit
from autograd import grad, jacobian

import sympy
from sympy.functions import exp

# 1. R-squared
# 2. Cook's Distance
# 3. 


###########################################################
#                                                         #
###########################################################

def model(bvals, params):
    """
    """

    d = params[0]
    pd = params[1]
    f = params[2]

    #print(d)
    #print(pd)
    #print(f)

    #print((1-f)*np.exp(-bvals*d)+f*np.exp(-bvals*pd))

    return (1-f)*np.exp(-bvals*d)+f*np.exp(-bvals*pd)


def residuals(bvals, params, data):
    ''' Implement Residuals Calc

        Parameters
	----------    
        Returns
    	-------
        References
    	----------
       	[1] - 
    '''
    
    #print(data.shape)
    #print(model(bvals, params))
    
    return data - model(bvals, params)


def rsquared(bvals, data, params):
    """

    Params:
        - data
        - params
    """
    
    if(len(np.array(data).shape) == 1):

        y_mean = np.mean(data)
        ssres = sum(residuals(bvals, params, data)**2)
        sstot = sum((data - y_mean)**2)

        rsqd = 1 - ssres/sstot

        return rsqd        
    
    else:
        
        d, d_star, f = params
        
        rsqd = []

        for i in range(data.shape[0]):    
            for j in range(data.shape[1]):    
                for k in range(data.shape[2]):
                    y_mean = np.mean(data[i,j,k])
                    ssres = sum(residuals(bvals, [d[i,j,k],d_star[i,j,k],f[i,j,k]], data[i,j,k])**2)
                    sstot = sum((data[i,j,k] - y_mean)**2)
                    
                    rsqd.append(1 - ssres/sstot)

        rsqd_mean = np.mean(rsqd)
        
        return rsqd_mean
        
###########################################################
#                                                         #
###########################################################


def gradient(bvals, d, d_star, pfraction):
    ''' Implement gradient of the function
        Parameters
    ----------
        Returns
    -------
        References
    ----------
        [1] - 
    '''
    
    col1 = -np.exp(-bvals*d)+np.exp(-bvals*d_star)
    col2 = -(1-pfraction)*bvals*np.exp(-bvals*d)
    col3 = -pfraction*bvals*np.exp(-bvals*d_star)

    Gradient = np.matrix([col1, col2, col3]).T

    return Gradient


def Hmatrix(gradient):
    ''' Implement Cook Distance for linear case
        Parameters
    ----------
        Returns
    -------
        References
    ----------
        [1] - 
    '''
    try:
        H = gradient @ np.linalg.inv(gradient.T @ gradient) @ gradient.T
        return H
    except np.linalg.LinAlgError:
        print("erro")
        H = 0
        return H

def rstudentized(S, bvals, d, d_star, pfraction):
    ''' Implement Cook Distance for linear case
        Parameters
    ----------
    
        Returns
    -------
    
        References
    ----------
        [1] - 
    '''
    
    S = S/S[0]

    params = [d, d_star, pfraction]

    #Calculate ri (residual)
    ri = residuals(bvals, params, S)

    #Calculate sigma ()
    sig = np.std(ri)

    #Calculate gradiente ()
    grad = gradient(bvals, d, d_star, pfraction)

    #Calculate hii ()
    h = Hmatrix(grad)

    if(type(h) == int):
        return 0
    else:
        rst = np.zeros(len(h))
        for i in range(0,len(h)):
            rst[i]=ri[i]/(sig*np.sqrt(1-h[i,i]))
    
        return rst

def cooksdistance(p, bvals, data, params, m):
    ''' Implement Cook Distance for linear case
        Parameters
    ----------
        Returns
    -------
        References
    ----------
        [1] - 
    '''
    
    if(len(np.array(data).shape) == 1):

        d, d_star, pfraction = params
        
        data_voxel = data/data[0]
    
        #Calculate gradiente ()
        grad = gradient(bvals, d, d_star, pfraction)

        #Calculate h ()
        h = Hmatrix(grad)
        
        if(type(h) == int):
            return 0
        else:        
            #Calculate t (studentized residual)
            t = rstudentized(data_voxel, bvals, d, d_star, pfraction)
        
            print(h)
            print(t)
        
            cooksdistancelinear = np.zeros(len(bvals))

            #Calculate Cooksdistance
            for i in range(len(bvals)):
                cooksdistancelinear[i] = (t[m]**2/p)*(h[m,m]/(1-h[m,m]))

        return cooksdistancelinear     
    
    else:
        
        d, d_star, pfraction = params
        
        cooksdistancel = np.zeros(len(bvals))

        data_voxel = data[i,j,k]/max(data[i,j,k])
        print(data_voxel)
        print(params)
        #Calculate gradiente ()
        grad = gradient(bvals, d, d_star, pfraction)

        #Calculate h ()
        h = Hmatrix(grad)

        #Calculate t (studentized residual)
        t = rstudentized(data_voxel, bvals, d, d_star, pfraction)

        cooksdistancelinear = np.zeros(len(bvals))

        #Calculate Cooksdistance
        for i in range(len(bvals)):
            cooksdistancelinear[i] = (t[i]**2/p)*(h[i,i]/(1-h[i,i]))
        
        cooksdistance = [cooksdistance[i]+cooksdistancelinear[i] for i in range(len(bvals))]

        return cooksdistance
