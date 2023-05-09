import numpy as np
from mantid.simpleapi import * 
import MDUtils as mdu

def mask_QE_box_MD(input_md,qlim,elim):
    #given limits in Q, E, masks an MDHistoworkspace. Useful for masking regions with bad data.
    working_MD = input_md.clone()
    dims = working_MD.getNonIntegratedDimensions()
    q = mdu.dim2array(dims[0])
    e = mdu.dim2array(dims[1])
    I = np.copy(working_MD.getSignalArray())
    err = np.sqrt(np.copy(working_MD.getErrorSquaredArray()))
    qmin=qlim[0]
    qmax=qlim[1]
    emin=elim[0]
    emax=elim[1]

    for i in range(len(I[:,0])):
        for j in range(len(I[0])):
            point = I[i,j]
            q_curr = q[i]
            e_curr = e[j]
            if q_curr<=qmax and q_curr>=qmin and e_curr<=emax and e_curr>=emin :
                #In the 'box'
                I[i,j]=np.nan
                #Don't touch the error- should be effectively inf but we'll leave it alone.
                err[i,j]=np.inf
    working_MD.setSignalArray(I)
    working_MD.setErrorSquaredArray(err**2)
    return working_MD