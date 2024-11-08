import numpy as np  
import matplotlib.pyplot as plt
import MDUtils as mdu
from bin1D import *


def cut_MDHisto_powder(workspace_cut1D,axis,extents,integration_range, auto_plot=False, extra_text='',plot_min=0,plot_max=1e-4,debug=False):
    #Takes an MD Histo and returns x, y for a cut in Q or E
    #only for powder data
    # Workspace - an mdhistoworkspace
    # axis - |Q| or DeltaE (mev)
    # extents - min, max, step_size for cut axis- array
    # Integration range- array of min, max for axis being summed over

    #Normalize by num events
    sets = [workspace_cut1D]
    intensities = np.copy(workspace_cut1D.getSignalArray())*1.

    errors = np.sqrt(np.copy(workspace_cut1D.getErrorSquaredArray()*1.))
    #Normalize to events 
    events = np.copy(workspace_cut1D.getNumEventsArray())
    intensities/=events 
    errors/=events
    errors[np.isnan(intensities)]=1e30
    #clean of nan values
    intensities[np.isnan(intensities)]=0
    if debug==True:
        print('random row of intensities')
        print(intensities[3])

    dims = workspace_cut1D.getNonIntegratedDimensions()
    q = mdu.dim2array(dims[0])
    e = mdu.dim2array(dims[1])

    if axis=='|Q|':
        #First limit range in E
        e_slice = intensities[:,np.intersect1d(np.where(e>=integration_range[0]),np.where(e<=integration_range[1]))]
        slice_errs = errors[:,np.intersect1d(np.where(e>=integration_range[0]),np.where(e<=integration_range[1]))]
        #Integrate over E for all values of Q
        integrated_intensities = []
        integrated_errs=[]
        for i in range(len(e_slice[:,0])):
            q_cut_vals = e_slice[i]
            q_cut_err = slice_errs[i]

            q_cut_err=q_cut_err[np.intersect1d(np.where(q_cut_vals!=0)[0],np.where(~np.isnan(q_cut_vals)))]
            q_cut_vals=q_cut_vals[np.intersect1d(np.where(q_cut_vals!=0)[0],np.where(~np.isnan(q_cut_vals)))]


            if len(q_cut_vals>0):
                integrated_err=np.sqrt(np.nansum(q_cut_err**2))/len(q_cut_vals)
                integrated_intensity=np.average(q_cut_vals,weights=1.0/q_cut_err)
                integrated_errs.append(integrated_err)
                integrated_intensities.append(integrated_intensity)
            else:
                integrated_err=0
                integrated_intensity=0
                integrated_errs.append(integrated_err)
                integrated_intensities.append(integrated_intensity)

        q_vals = q
        binned_intensities = integrated_intensities
        binned_errors = integrated_errs
        bin_x = q_vals
        bin_y = binned_intensities
        bin_y_err = binned_errors
        other = '$\hbar\omega$'
        # Now bin the cut as specified by the extents array
        extent_res = np.abs(extents[1]-extents[0])
        bins = np.arange(extents[0],extents[1]+extents[2]/2.0,extents[2])
        bin_x,bin_y,bin_y_err = bin_1D(q,bin_y,bin_y_err,bins,statistic='mean')
    elif axis=='DeltaE':
        #First restrict range across Q
        q_slice = intensities[np.intersect1d(np.where(q>=integration_range[0]),np.where(q<=integration_range[1]))]
        slice_errs = errors[np.intersect1d(np.where(q>=integration_range[0]),np.where(q<=integration_range[1]))]
        #Integrate over E for all values of Q
        integrated_intensities = []
        integrated_errs=[]
        for i in range(len(q_slice[0])):
            e_cut_vals = q_slice[:,i]
            e_cut_err = slice_errs[:,i]
            e_cut_err = e_cut_err[np.intersect1d(np.where(e_cut_vals!=0)[0],np.where(~np.isnan(e_cut_vals)))]
            e_cut_vals=e_cut_vals[np.intersect1d(np.where(e_cut_vals!=0)[0],np.where(~np.isnan(e_cut_vals)))]


            if len(e_cut_vals)>0:
                integrated_err=np.sqrt(np.nansum(e_cut_err**2))/len(e_cut_vals)
                integrated_intensity=np.average(e_cut_vals,weights=1.0/e_cut_err)
                integrated_errs.append(integrated_err)
                integrated_intensities.append(integrated_intensity)
            else:
                integrated_errs.append(0)
                integrated_intensities.append(0)
        bin_x = e
        bin_y = integrated_intensities
        bin_y_err = integrated_errs

        bins = np.arange(extents[0],extents[1]+extents[2]/2.0,extents[2])
        bin_x,bin_y,bin_y_err = bin_1D(e,bin_y,bin_y_err,bins,statistic='mean')
        other = '|Q|'
    else:
        print('Invalid axis option (Use \'|Q|\' or \'DeltaE\')')
        return False
    if auto_plot==True:
        #Attempts to make a plot of the cut. Limits and such will be off

        cut_fig, cut_ax = plt.subplots(1,1,figsize=(8,6))
        cut_ax.set_title(axis+' Cut')
        cut_ax.set_xlabel(axis)
        cut_ax.set_ylabel('Intensity (arb.)')
        cut_ax.errorbar(x=bin_x,y=bin_y,yerr=bin_y_err,marker='o',color='k',ls='--',mfc='w',mec='k',mew=1,capsize=3)
        cut_ax.set_ylim(np.nanmin(bin_x)*0.8,np.nanmax(bin_y)*1.5)
        cut_ax.text(0.7,0.95,other+'='+str(integration_range),transform=cut_ax.transAxes)
        cut_ax.text(0.7,0.85,extra_text,transform=cut_ax.transAxes)
        cut_fig.show()

    return np.array(bin_x),np.array(bin_y),np.array(bin_y_err)