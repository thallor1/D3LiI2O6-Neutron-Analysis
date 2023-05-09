from mdfactorization import MDfactorization
from mantidFF import get_MANTID_magFF
from progressbar import ProgressBar
from cut_mdhisto_powder import cut_MDHisto_powder
from subnearest import sub_nearest_MD
import MDUtils as mdu
import numpy as np
from mantid.simpleapi import *
import matplotlib.pyplot as plt 

def MDsubmultscatt(md, qrange_dos, erange_dos,Ei,T,norm_qrange,norm_erange,mag_ion='Ir4',sub_scale=1.0,factorizeDOS=True):
    #Performs specific analysis of multiple scattering using incoherent and inelastic process
    # Details of this analysis can be found elsewhere:
    #Input arguments
    #   md - MDHistoworkspace
    #   Qrange_dos - list in format [qlow, qhigh] used for extraction of DOS. Should be as high as possible.
    #   erange_dos - list in format of [elow,ehigh] used for extraction of DOS, should cut off just before elastic line
    #   Ei - incident neutron energy
    #   T - Effective transmission of neutrons that don't go through this process, extracted elsewhere. 
    #   norm_qrange - list for normalization of final multiple scattering subtraction in form of [qmin,qmax]
    #   norm_erange - same as previous but for energy - [emin,emax]

    # outputs a single MDHistoworkspace which is the lowT MDHisto - the multiple scattering
    qdos,f_q,f_q_err,edos,dos,doserr = MDfactorization(md,mag_ion,Ei=Ei,twoThetaMin=3.5,plot_result=False,
                                                                         fname='mult_scatt_low_Ei_'+str(Ei)+'_q_'+str(qrange_dos)+'_e_'+str(erange_dos)+'_uncertainties.txt',q_lim=qrange_dos,e_lim=erange_dos,fast_mode=True,overwrite_prev=True)
    dims = md.getNonIntegratedDimensions()
    q = mdu.dim2array(dims[0])
    e = mdu.dim2array(dims[1])
    ecut,icut,errcut = cut_MDHisto_powder(md,'DeltaE',[erange_dos[0],erange_dos[1],np.abs(e[1]-e[0])],qrange_dos)
    #Make sure ratios of integrated intensities of factorizations is equal to that of cuts
    #This is because factorization is not defined in terms of direct intensities. 
    cut_integral = np.trapz(icut,x=ecut)
    fact_integral = np.trapz(dos,x=edos)

    ratio = fact_integral/ cut_integral

    dos_norm = dos / ratio

    #sometimes factorizaiton isn't smooth: can use the cut instead.
    if factorizeDOS==False:
        dos_norm=icut

    q=mdu.dim2array(dims[0])
    Q,E = np.meshgrid(q,edos)
    T = T
    kb = 8.617e-2
    lambda_i = np.sqrt(81.81 / (Ei))
    lambda_f =np.sqrt(81.81 / (Ei-E))
    ki = 2.0*np.pi/lambda_i
    kf = 2.0*np.pi/lambda_f
    kisqr_kfsqr = (ki**2 + kf**2)
    A = kisqr_kfsqr*(T*(Q**2/(kisqr_kfsqr))+(1.0-T))
    dos_mesh = np.zeros(np.shape(Q))
    for i in range(len(dos_mesh[0])):
        dos_mesh[:,i]=A[:,i]*dos_norm


    extents_str = str(np.min(q))+','+str(np.max(q))+','+str(np.min(edos))+','+str(np.max(edos))
    num_bin_str = str(len(np.unique(q)))+','+str(len(np.unique(edos)))
    I_arr=dos_mesh.flatten()
    err_arr = np.ones(len(I_arr))
    #Below is the calculated phonon + multiple scattering, or I'(Q,omega)
    calcMD= CreateMDHistoWorkspace(Dimensionality=2,Extents=extents_str,SignalInput=I_arr,\
                                        ErrorInput=err_arr,NumberOfBins=num_bin_str,NumberOfEvents=np.ones(len(I_arr))\
                                        ,Names='|Q|,DeltaE',Units='MomentumTransfer,\
                                        EnergyTransfer')


    eres = np.abs(e[1]-e[0])
    #Normalize these such that a high energy, high Q cut subtracts to zero
    em,im,errm = cut_MDHisto_powder(md,'DeltaE',[norm_erange[0],norm_erange[1],eres*2.0],norm_qrange)
    eC,iC,errC = cut_MDHisto_powder(calcMD,'DeltaE',[norm_erange[0],norm_erange[1],eres*2.0],norm_qrange)
    int = np.trapz(im,x=em)
    int_calc = np.trapz(iC,x=eC)
    scale = int/int_calc
    print(f"Scale={scale}")
    normMD = calcMD*scale
    #Set the error bars on the calc to zero
    norm_err = normMD.getErrorSquaredArray()
    norm_err = np.zeros(np.shape(norm_err))
    normMD.setErrorSquaredArray(norm_err)
    pre_err = md.getErrorSquaredArray()
    #Rescale if a scale factor was defined
    normMD = normMD*sub_scale
    ####

    Ibar = sub_nearest_MD(md,normMD)
    Ibar = Ibar.clone()
    plt.show()
    Ibar.setErrorSquaredArray(pre_err)
    return Ibar, normMD