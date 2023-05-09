#General class for handling of iexy files, mostly uneeded 
import numpy as np  
import matplotlib.pyplot as plt 
import copy
from mantid.simpleapi import *

def load_iexy(fnames):
    #Sequentially loads IEXY files from DAVE, appending data to the last one.
    # Takes either a str or a list of strs
    if type(fnames)==str:
        #single fname, simple.
        iexy = np.genfromtxt(fnames)
    elif type(fnames)==list:
        #multiple filenames
        iexy = np.empty((0,4))
        for i in range(len(fnames)):
            iexy = np.vstack((iexy,np.genfromtxt(fnames[i])))
    return iexy

class IEXY_data:
    '''
    Class for handling iexy data form DAVE
    '''
    def __init__(self,fnames=0,scale_factor=1,Ei=5.0,self_shield=1.0):
        if fnames:
            iexy = load_iexy(fnames)
            self.intensity = iexy[:,0]*scale_factor
            self.err = iexy[:,1]*scale_factor
            self.q = iexy[:,2]
            self.energies = iexy[:,3]
            self.Ei=Ei


    def delete_indices(self,indices):
        #Tool to mask pixels in IEXY. Useful for
        # Masking or stitching
        indices=np.unique(indices)
        self.intensity = np.delete(self.intensity,indices)
        self.err = np.delete(self.err,indices)
        self.q = np.delete(self.q,indices)
        self.energies = np.delete(self.energies,indices)

    def get_indices(self,indices):
        #gets data from specified indices of object
        intensity=self.intensity[indices]
        err = self.err[indices]
        q = self.q[indices]
        energies = self.energies[indices]
        return intensity,err,q,energies

    def scale_iexy(self,scale_factor):
        #Scales the data to a precomputed scale factor
        self.intensity = self.intensity*scale_factor
        self.err=self.err*scale_factor

    def sub_bkg(self,bkg_iexy,self_shield=1.0):
        #subtracts another iexy from this one using the nearest value
        # in the background IEXY
        # TODO- implement interpolation rather than neearest neighbor
        for i in range(len(self.intensity)):
            closest_arg = np.argmin(np.abs(bkg_iexy.q-self.q[i])+np.abs(bkg_iexy.energies-self.energies[i]))
            self.intensities[i]=self.intensitiy[i]-self_shield*bkg_iexy.intensity[closest_arg]
            self.err[i]=np.sqrt(self.err[i]**2 + (self_shield*bkg_iexy.err[closest_arg])**2 )
    def bose_subtract(self,highT_iexy,tlow,thigh):
        #Performs bose-einstein temperature subtraction using a highT iexy dataset
        self.intensity= self.intensity

    def normalize_to_bragg(self,ref_iexy,res_Q,res_E,ref_Q_res,ref_E_res,bragg_I,bragg_I_ref):
        #Normalizes one dataset to another using bragg peak intensities
        # Requires resolution in Q, E for both datasets
        # Assumes reference dataset is already normalized and that both have
        # been adjusted for energy-dependent transmission.
        # Should use something like DAVE to get the intensity of the peak normalized to the same monitor
        scale_factor = (bragg_I_ref*ref_E_res*ref_Q_res)/(bragg_I*E_res*Q_res)
        self.intensity*=scale_factor
        self.err*=scale_factor

    def take_q_cut(self,q_range,e_range,plot=True):
        #Simple function to take a cut of an iexy object
        #Q range in form of [min,max,num_bins]
        #E range in form of [min,max]
        #returns Q,I(Q),Err(Q)
        q_cut_i = np.intersect1d(np.where(self.q>=q_range[0]),np.where(self.q<=q_range[1]))
        e_slice_i = np.intersect1d(np.where(self.energies>=e_range[0]),np.where(self.energies<=e_range[1]))
        all_i = np.unique(np.append(q_cut_i,e_slice_i))
        slice_I, slice_err, slice_Q, slice_E = self.get_indices(all_i)
        bin_edges = np.linspace(q_range[0],q_range[1],q_range[2]+1)
        q_cut =[]
        q_cut_err=[]
        q_bin_centers=[]
        for i in range(len(bin_edges)-1):
            ind_in_bin = np.intersect1d(np.where(slice_Q>bin_edges[i]),np.where(slice_Q[i]<bin_edges[i]))
            cut_val = np.mean(slice_I[ind_in_bin])
            cut_err = np.sqrt(np.sum(slice_err[ind_in_bin]**2))/len(ind_in_bin)
            q_cut.append(cut_val)
            q_cut_err.append(cut_err)
            q_bin_centers.append(np.mean([bin_edges[i],bin_edges[i+1]]))
        if plot==True:
            plt.figure()
            plt.title('Q-cut from E=['+str(e_range[0])+','+str(e_range[1])+']')
            plt.xlabel('|Q|$\AA^{-1}$')
            plt.ylabel('Intensity (arb.)')
            plt.xlim(q_range[0],q_range[1])
            plt.ylim(np.min(q_cut),np.median(q_cut)*4.0)
            plt.show()
        return q_bin_centers,q_cut,q_cut_err

    def absorb_correct(self,rho_abs,vol,num_formula_units=False,d=1.0):
        #Corrects dataset for energy dependent absorption
        #d is path traveled in cm
        # rho is total absorption cross section of material
        # vol is the unit cell volume
        # num_formula_units is the number of formula units in the unit cell
        ref_lambda= 3956.0/2200.0
        lambda_i = 9.045 / np.sqrt(self.Ei)
        lambda0=ref_lambda
        energies_f = self.Ei - self.energies
        lambda_f = 9.045/np.sqrt(energies_f)
        if num_formula_units==False:
            print('WARNING: Number of formula units per unit cell must be specified for this calculation. ')
            return False
        for i in range(len(self.intensity)):
            ratio = (lambda_i + lambda_f[i])/(2.0*lambda0)
            transmission = np.exp(-0.5*d*rho_abs*ratio/vol)
            self.intensity[i]=self.intensity[i]/transmission
            self.err[i]=self.err[i]/transmission

    def take_cut(self,cut_axis='x',cut_params=[0,10,0.1],integrated_axis_extents=[0,1],plot=True):
        #Simple function to take a cut of an iexy object
        #  Define if the axis being cut is X or Y
        #  cut_params in format of [min,max,resolution]
        #  integrated_axis_extents defines integration region of other axis [min,max

        I_all = self.intensity 
        x_all = self.q 
        y_all = self.energies
        err_all = self.err
        if cut_axis=='x':
            integrated_i = np.intersect1d(np.where(y_all<=integrated_axis_extents[1]),np.where(y_all>=integrated_axis_extents[0]))
        elif cut_axis=='y':
            integrated_i = np.intersect1d(np.where(x_all<=integrated_axis_extents[1]),np.where(x_all>=integrated_axis_extents[0]))
        else:
            print('Invalid cut axis argument- only x or y permitted.')
            return 0
        I_all = I_all[integrated_i]
        x_all = x_all[integrated_i]
        y_all = y_all[integrated_i]
        err_all = err_all[integrated_i]
        #Integrate the relavant axis and errors
        if cut_axis=='x':
            #sort points into x_bins, then integrate Y
            x_bins = np.arange(cut_params[0],cut_params[1]+cut_params[2]/2.0,cut_params[2])
            x = x_bins[1:]-(x_bins[1]-x_bins[0])/2.0
            y = np.zeros(len(x))
            err = np.zeros(len(x))
            for i in range(len(x_bins)-1):
                ind = np.intersect1d(np.where(x_all>=x_bins[i]),np.where(x_all<=x_bins[i+1]))
                bin_errs = err_all[ind]
                bin_I = I_all[ind]
                if len(ind)>0:
                    y[i] = np.average(I_all[ind],weights=1.0/bin_errs)
                    err[i]=np.sqrt(np.sum(bin_errs**2))/len(bin_errs)
        elif cut_axis=='y':
            #sort points into x_bins, then integrate Y
            x_bins = np.arange(cut_params[0],cut_params[1]+cut_params[2]/2.0,cut_params[2])
            x = x_bins[1:]-(x_bins[1]-x_bins[0])/2.0
            y = np.zeros(len(x))
            err = np.zeros(len(x))
            for i in range(len(x_bins)-1):
                ind = np.intersect1d(np.where(y_all>=x_bins[i]),np.where(y_all<=x_bins[i+1]))
                bin_errs = err_all[ind]
                bin_I = I_all[ind]
                if len(ind)>0:
                    y[i] = np.average(I_all[ind],weights=1.0/bin_errs)
                    err[i]=np.sqrt(np.sum(bin_errs**2))/len(bin_errs)
        #If the user chose too fine a resolution there will be zero bins- remove these
        bad_bins=np.where(y==0)[0]
        x = np.array(x)
        y=np.array(y)
        err=np.array(err)
        x = np.delete(x,bad_bins)
        y = np.delete(y,bad_bins)
        err = np.delete(err,bad_bins)

        return x,y,err

    def rebin_iexy(self,x_bins,y_bins,return_new=True):
        #Given arrays for x and y bin edges, rebins the dataset appropriately.
        # If return new is set to true, returns a new object
        # If changed to false, edits the current object
        x_res = np.abs(x_bins[0]-x_bins[1])/2.0
        y_res = np.abs(y_bins[1]-y_bins[0])/2.0
        x_bin_centers = x_bins[1:]-x_res 
        y_bin_centers = y_bins[1:]-y_res 
        I_new=[]
        err_new =[]
        x_new =[]
        y_new =[]
        for i in range(len(x_bins)-1):
            for j in range(len(y_bins)-1):
                #find all intensities that lie in the bin
                xmin=x_bins[i]
                xmax=x_bins[i+1]
                ymin=y_bins[j]
                ymax=y_bins[j+1]
                x_ind = np.intersect1d(np.where(self.q >=xmin),np.where(self.q<xmax))
                y_ind = np.intersect1d(np.where(self.energies >=ymin),np.where(self.energies<ymax))
                ind = np.intersect1d(x_ind,y_ind)
                if len(ind)>0:
                    I_arr = np.array(self.intensity[ind])
                    err_arr = np.array(self.err[ind])
                    zero_errs = np.where(err_arr==0)[0]
                    err_arr[zero_errs]=1e8
                    weights = 1.0/err_arr
                    weights[zero_errs]=0.0 
                    if np.sum(weights)==0:
                        weights=np.ones(len(weights))
                    I_bin=np.average(I_arr,weights=weights)
                    err_bin = np.sqrt(np.sum(err_arr**2))/len(err_arr)
                    x_new.append(x_bin_centers[i])
                    y_new.append(y_bin_centers[j])
                    I_new.append(I_bin)
                    err_new.append(err_bin)
                else:
                    x_new.append(x_bin_centers[i])
                    y_new.append(y_bin_centers[j])
                    I_new.append(np.nan)
                    err_new.append(np.nan)
        if return_new==True:
            copy_obj = copy.deepcopy(self)
            copy_obj.q=np.array(x_new)
            copy_obj.energies = np.array(y_new)
            copy_obj.intensity = np.array(I_new)
            copy_obj.err = np.array(err_new)
            return copy_obj
        else:
            self.q=np.array(x_new) 
            self.energies = np.array(y_new)
            self.intensity = np.array(I_new)
            self.err=np.array(err_new)



    def transform2D(self,q_decimals=4,e_decimals=8):
        #Good for cleaning up dataset, bins data in Q and E
        # returns grid objects suitable for pcolormesh
        Q_arr = np.sort(np.around(np.unique(self.q),q_decimals))
        E_arr = np.sort(np.around(np.unique(self.energies),e_decimals))
        Q_grid,E_grid = np.meshgrid(Q_arr,E_arr)
        I_grid,err_grid = np.zeros(np.shape(Q_grid)),np.zeros(np.shape(Q_grid))
        for i in range(len(self.intensity)):
            q = self.q[i]
            e = self.energies[i]
            Int = self.intensity[i]
            err = self.err[i]
            q_index = np.argmin(np.abs(Q_arr-q))
            e_index = np.argmin(np.abs(E_arr-e))
            I_grid[e_index,q_index]=Int
            err_grid[e_index,q_index]=err
        return Q_grid,E_grid,I_grid,err_grid

    def plot_slice(self,axis_extents=False,vmin=0,vmax=1e4,cmap='rainbow',title='IEXY Slice plot',xlabel='|Q|',ylabel='E'):
        #Plots a slice of the dataset, returns the figure and ax 
        # axes extents in form of [xmin,xmax,ymin,ymax]
        if axis_extents==False:
            axis_extents=[-1e10,1e10,-1e10,1e10]
        fig,ax=plt.subplots(1,1)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        X,Y,I,E = self.transform2D()
        #Z = scipy.interpolate.griddata((x_arr,y_arr),z_arr,(X,Y),method='linear')
        cax=fig.add_axes([1.0,0.1,0.05,0.75])
        mesh=ax.pcolormesh(X,Y,I,vmin=vmin,vmax=vmax,cmap=cmap)
        fig.colorbar(mesh,cax=cax,orientation='vertical')
        plt.show()
        return fig,ax

    def scale_to_FF(self,mag_ion):
        #Scales to the catalogued MANTID magneitc form factor
        #Assumes that the x is Q, y is E (i.e. powder)
        q_arr = self.q 
        q_min = np.min(q_arr)
        q_max = np.max(q_arr)
        q_all = np.linspace(q_min,q_max,1000)
        Q_ff, FF = THfuncs.get_MANTID_magFF(q_all,mag_ion)
        for i in range(len(self.intensity)):
            q_ind = np.argmin(np.abs(q_all-self.q[i]))
            FF_ind = FF[q_ind]
            self.intensity[i]*=FF_ind
            self.err[i]*=FF_ind 

    def save_IEXY(self,fname):
        #Saves manipulated data to a new file for later
        q_arr = np.array(self.q)
        e_arr = np.array(self.energies)
        I_arr = np.array(self.intensity)
        err_arr = np.array(self.err)
        mat = np.array([I_arr,err_arr,q_arr,e_arr])
        np.savetxt(fname,mat.T)

    def convert_to_MD(self):
        #Converts the IEXY to an MDhisto workspace for use with other algorithms.
        q_arr = np.array(self.q)
        e_arr = np.array(self.energies)
        I_arr = np.array(self.intensity)
        err_arr = np.array(self.err)
        err_arr[np.isnan(I_arr)]=0
        I_arr[np.isnan(I_arr)]=0
        #Need to do the sorting systematically. First get all elements in the lowest energy bin
        #First get all elements in the lowest E-bin
        new_I_arr=[]
        new_err_arr=[]
        for i in range(len(np.unique(np.around(e_arr,3)))):
            e_val=np.unique(np.around(e_arr,3))[i]
            e_bin_indices = np.where(np.around(e_arr,3)==e_val)[0]
            curr_q_arr = q_arr[e_bin_indices]
            curr_I_arr=I_arr[e_bin_indices]
            curr_err_arr=err_arr[e_bin_indices]
            q_sorted = np.argsort(curr_q_arr)
            new_I_arr.append(curr_I_arr[q_sorted])
            new_err_arr.append(curr_err_arr[q_sorted])
        I_arr=np.array(new_I_arr)
        err_arr=np.array(new_err_arr)
        extents_str = str(np.min(self.q))+','+str(np.max(self.q))+','+str(np.min(self.energies))\
                        +','+str(np.max(self.energies))
        num_bin_str = str(len(np.unique(self.q)))+','+str(len(np.unique(self.energies)))
        
        out_ws = CreateMDHistoWorkspace(Dimensionality=2,Extents=extents_str,SignalInput=I_arr,ErrorInput=err_arr,NumberOfBins=num_bin_str,NumberOfEvents=np.ones(len(self.intensity)),Names='Dim1,Dim2',Units='MomentumTransfer,EnergyTransfer')
        return out_ws