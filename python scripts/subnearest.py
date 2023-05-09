import MDUtils as mdu  
import numpy as np 
def sub_nearest_MD(md_left,md_right,mode='subtract'):
    #allows for subtraction of MDHistos with unequal bin sizes. Uses closest value in coarser grid.
    out_MD = md_left.clone()
    dims = out_MD.getNonIntegratedDimensions()
    qLeft = mdu.dim2array(dims[0])
    eLeft = mdu.dim2array(dims[1])

    sub_dims = md_right.getNonIntegratedDimensions()
    q_sub = mdu.dim2array(sub_dims[0])
    e_sub = mdu.dim2array(sub_dims[1])
    events_left = np.copy(md_left.getNumEventsArray())
    events_right = np.copy(md_right.getNumEventsArray())
    I_Left=np.copy(md_left.getSignalArray())
    I_new = np.copy(I_Left)#/events_left
    Ierr_left = np.sqrt(np.copy(md_left.getErrorSquaredArray()))#/events_left
    new_err = np.copy(Ierr_left)
    I_sub = np.copy(md_right.getSignalArray())#/events_right
    I_sub_err = np.sqrt(np.copy(md_right.getErrorSquaredArray()))#/events_right
    for i in range(len(qLeft)):
        for j in range(len(eLeft)):
            q_arg = np.argmin(np.abs(q_sub-qLeft[i]))
            e_arg = np.argmin(np.abs(e_sub-eLeft[j]))
            I_new[i,j]=I_Left[i,j]-I_sub[q_arg,e_arg]

            err_sub = I_sub_err[q_arg,e_arg]
            err_net = np.sqrt(Ierr_left[i,j]**2 + err_sub**2)
            new_err[i,j]=err_net 
    out_MD.setSignalArray(I_new)
    out_MD.setErrorSquaredArray(new_err**2)
    return out_MD