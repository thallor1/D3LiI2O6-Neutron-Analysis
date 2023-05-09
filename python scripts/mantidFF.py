from mantid.simpleapi import *
import numpy as np

def get_MANTID_magFF(q,mag_ion):
    #Given a str returns a simple array of the mantid defined form factor. basically a shortcut for the mantid version
    cw = CreateWorkspace(DataX = q,DataY = np.ones(len(q)))
    cw.getAxis(0).setUnit('MomentumTransfer')
    ws_corr = MagFormFactorCorrection(cw,IonName=mag_ion,FormFactorWorkspace='FF')
    FFcorrection = np.array(ws_corr[0].readY(0))
    return q,FFcorrection