from mantid.simpleapi import *
import numpy as np

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
        events = np.copy(cut2D.getNumEventsArray())
        I = np.copy(cut2D.getSignalArray())
        Err = np.sqrt(np.copy(cut2D.getErrorSquaredArray()))
        I/=events
        Err/=events
        cut2D.setSignalArray(I)
        cut2D.setErrorSquaredArray(Err**2)
    cut2D*=van_factor
    return cut2D

def NxspeToMD(file,van_factor=1.0):
    # Takes file array and turns it into mdhisto
    matrix_ws = LoadNXSPE(file)
    #Convert to MD
    md=ConvertToMD(matrix_ws,Qdimensions='|Q|')
    md*=van_factor
    return md