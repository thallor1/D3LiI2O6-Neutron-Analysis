import numpy as np  
from mantid.simpleapi import * 
import MDUtils as mdu  
from subnearest import sub_nearest_MD

def tempsubtract_nearest_cut2D(md_low,md_high,t_low,t_high):
    #Helper function to do a bose-einstein subtraction of MDHistodatasets with nearest-neighbor values (when the dims don't match exactly)
    high_I = np.copy(md_high.getSignalArray())
    dims = md_high.getNonIntegratedDimensions()
    q = mdu.dim2array(dims[0])
    e = mdu.dim2array(dims[1])
    Q,E = np.meshgrid(q,e)
    kb = 8.16e-2
    bose_fact = (1.0-np.exp(-E/(kb*55.0)))/(1.0-np.exp(-E/(kb*2.0)))
    bose_fact = np.transpose(bose_fact)
    high_I *=bose_fact
    high_MD_scaled= md_high.clone()
    high_MD_scaled.setSignalArray(high_I)
    cut2D_mag_tempsub=sub_nearest_MD(md_low,high_MD_scaled).clone()
    return cut2D_mag_tempsub