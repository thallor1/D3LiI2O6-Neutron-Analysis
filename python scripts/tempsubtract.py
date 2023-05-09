from mantid.simpleapi import *
import MDUtils as mdu 
import numpy as np 
def tempsubtract_cut2D(lowT_cut2D_T,highT_cut2D_T,tLow,tHigh,numEvNorm=False,vmin=0,vmax=5):
    #Same as normal bose einstein temperature subtraction but with cut2d workspaces instead of filenames.
    #Normalize by num events
    #Scale the highT dataset by bose-population factor
    highT_cut2D_tempsub = highT_cut2D_T.clone()
    lowT_cut2D_tempsub = lowT_cut2D_T.clone()
    hight_plot_tempsub = highT_cut2D_tempsub.clone()
    #pre_H_fig,pre_H_ax = fancy_plot_cut2D(hight_plot_tempsub, vmin=0,vmax=10,title='T=120K pre-scale')

    dims = lowT_cut2D_tempsub.getNonIntegratedDimensions()
    q_values = mdu.dim2array(dims[0])
    energies = mdu.dim2array(dims[1])
    if numEvNorm==True:
        lowT_cut2D_tempsub=normalize_MDHisto_event(lowT_cut2D_tempsub)
        highT_cut2D_tempsub = normalize_MDHisto_event(highT_cut2D_tempsub)
    kb=8.617e-2
    bose_factor_lowT = (1-np.exp(-energies/(kb*tLow)))
    bose_factor_highT = (1-np.exp(-energies/(kb*tHigh)))
    #Only makes sense for positive transfer
    bose_factor_lowT[np.where(energies<0)]=0
    bose_factor_highT[np.where(energies<0)]=0
    highT_Intensity = np.copy(highT_cut2D_tempsub.getSignalArray())
    highT_err = np.sqrt(np.copy(highT_cut2D_tempsub.getErrorSquaredArray()))
    bose_factor = bose_factor_highT/bose_factor_lowT
    highT_Intensity_corrected = bose_factor*highT_Intensity
    highT_err_corrected = bose_factor*highT_err
    highT_Intensity_corrected[np.where(highT_Intensity_corrected==0)]=0
    highT_err_corrected[np.where(highT_err_corrected==0)]=0
    highT_Intensity_corrected[np.isnan(highT_Intensity_corrected)]=0
    highT_err_corrected[np.isnan(highT_err_corrected)]=0

    highT_cut2D_tempsub.setSignalArray(highT_Intensity_corrected)
    highT_cut2D_tempsub.setErrorSquaredArray(highT_err_corrected**2)
    highT_plot_cut2D = highT_cut2D_tempsub.clone()
    lowt_tempsub_plot = lowT_cut2D_tempsub.clone()
    #pre_L_fig, pre_L_ax = fancy_plot_cut2D(lowt_tempsub_plot,vmin=vmin,vmax=vmax,title='T=5K')
    #post_H_fig, post_H_ax = fancy_plot_cut2D(highT_plot_cut2D,vmin=vmin,vmax=vmax,title='T=120K post-scale')
    #Don't really know if MANTID handles the subtraction well...
    lowT_cut2D_intensity = np.copy(lowT_cut2D_tempsub.getSignalArray())
    lowT_cut2D_err = np.sqrt(np.copy(lowT_cut2D_tempsub.getErrorSquaredArray()))

    mag_intensity = lowT_cut2D_intensity - highT_Intensity_corrected
    mag_err = np.sqrt(lowT_cut2D_err**2 + highT_err_corrected**2)

    cut2D_mag_tempsub= lowT_cut2D_tempsub.clone()
    cut2D_mag_tempsub.setSignalArray(mag_intensity)
    cut2D_mag_tempsub.setErrorSquaredArray(mag_err**2)
    cut2D_mag_tempsub_plot = cut2D_mag_tempsub.clone()
    #manget_fig,magnet_ax = fancy_plot_cut2D(cut2D_mag_tempsub_plot,vmin=vmin,vmax=vmax,title='magnetism')
    return cut2D_mag_tempsub