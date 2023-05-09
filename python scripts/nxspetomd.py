from mantid.simpleapi import *
import numpy as np
import MDUtils as mdu

def NxspeToMDHisto(file,Qbins='[0,3,0.1]',Ebins='[-10,10,0.1]',numEvNorm=True,van_factor=1.0):
    # Takes file array and turns it into mdhisto
    matrix_ws = LoadNXSPE(file)
    #Convert to MD
    md=ConvertToMD(matrix_ws,Qdimensions='|Q|')
    #Bin both
    cut2D = BinMD(md,AxisAligned=True,AlignedDim0=Qbins,AlignedDim1=Ebins)
    #Normalize to event
    #Normalize by num events (already done earlier in an update)
    if numEvNorm==True:
        dims = cut2D.getNonIntegratedDimensions()
        q = mdu.dim2array(dims[0])
        e = mdu.dim2array(dims[1])
        events = np.copy(cut2D.getNumEventsArray()).T
        I = np.copy(cut2D.getSignalArray()).T
        Err = np.sqrt(np.copy(cut2D.getErrorSquaredArray())).T
        I/=events
        Err/=events
        Iarr=I.flatten()
        err_arr = Err.flatten()
        extents_str = str(np.min(q))+','+str(np.max(q))+','+str(np.min(e))\
                            +','+str(np.max(e))
        num_bin_str = str(len(np.unique(q)))+','+str(len(np.unique(e)))
        out2D = CreateMDHistoWorkspace(Dimensionality=2,Extents=extents_str,SignalInput=Iarr,\
                ErrorInput=err_arr,NumberOfBins=num_bin_str,NumberOfEvents=np.ones(len(Iarr))\
                ,Names='|Q|,DeltaE',Units='MomentumTransfer,EnergyTransfer')
        cut2D=out2D
    cut2D*=van_factor
    return cut2D

def NxspeToMD(file,van_factor=1.0):
    # Takes file array and turns it into mdhisto
    matrix_ws = LoadNXSPE(file)
    #Convert to MD
    md=ConvertToMD(matrix_ws,Qdimensions='|Q|')
    md*=van_factor
    return md