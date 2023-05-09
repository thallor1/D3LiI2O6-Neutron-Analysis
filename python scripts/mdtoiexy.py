from iexydata import IEXY_data
import numpy as np




def make_MD_from_iexy(iexy_fname,Ei=5.0):
    #given an iexy file, creates an MDHistoworkspace for manipulation using MANTID algorithms.
    iexy_obj = IEXY_data(iexy_fname,Ei=Ei)
    #Needs to be rebinned to be compatible
    e_arr = np.unique(iexy_obj.energies)
    q_arr = np.unique(iexy_obj.q)
    e_res = np.abs(e_arr[1]-e_arr[0])
    q_res = np.abs(q_arr[1]-q_arr[0])
    q_min = np.min(q_arr)-q_res 
    q_max = np.max(q_arr)+q_res 
    num_qbins = len(q_arr)
    e_min = np.min(e_arr)-e_res 
    e_max = np.max(e_arr)+e_res 
    num_ebins = len(e_arr)
    binned_IEXY = iexy_obj.rebin_iexy(np.linspace(q_min,q_max,num_qbins),np.linspace(e_min,e_max,num_ebins))

    out_MD = binned_IEXY.convert_to_MD()
    return_MD = out_MD.clone()
    return return_MD