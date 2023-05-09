import numpy as np
import MDUtils as mdu
from mantid.simpleapi import *

def minQseq(Ei,twoTheta,deltaE=0):
    #Returns lowest Q for Ei
    deltaEmax = Ei*0.9
    if deltaE==0:
        deltaE = np.linspace(0,deltaEmax,1000)

    Ef = Ei - deltaE

    ki = np.sqrt(Ei/2.07)
    kf = np.sqrt(Ef/2.07)
    Q = np.sqrt( ki**2 + kf**2 - 2*ki*kf*np.cos(twoTheta*np.pi/180))
    return Q, deltaE
def minQmacs(Ef,twoTheta,deltaE=0):
    #Returns the lowest available Q for a given Ef and energy transfer
    Ei = deltaE +Ef 
    ki = np.sqrt(Ei/2.07)
    kf = np.sqrt(Ef/2.07)
    Q = np.sqrt( ki**2 + kf**2 - 2*ki*kf*np.cos(twoTheta*np.pi/180.0))
    return Q, deltaE

def minQseq_multEi(Ei_arr,twoTheta,deltaE=0):
    #returns lowest accessible Q for a number of Ei's
    if not len(Ei_arr)>1:
        print('This function only takes an array of incident energies.') 
        return 0

    Eiarr=np.array(Ei_arr)
    if deltaE==0:
        deltaE=np.linspace(0,np.max(Ei_arr)*0.9,1000)
    Q_final_arr=[]
    #for every deltaE find which Ei has the lowest accessible Q. 
    # if the deltaE>0.9*Ef then this Ei is impossible
    for i in range(len(deltaE)):
        delE=deltaE[i]
        minQ=1000.0 #placeholder values
        for j in range(len(Eiarr)):
            Ei=Eiarr[j]
            if Ei>=0.9*delE:
                #allowed
                Ef = Ei - delE
                ki = np.sqrt(Ei/2.07)
                kf = np.sqrt(Ef/2.07)
                Q = np.sqrt( ki**2 + kf**2 - 2*ki*kf*np.cos(twoTheta*np.pi/180))
            else:
                Q=10.0
            if Q<minQ:
                minQ=Q
        Q_final_arr.append(minQ)
    return np.array(Q_final_arr),np.array(deltaE)

def mask_minQ_fixedEi_MD(seq_MD,twoThetaMin,Ei):
    #Remove areas outside of kinematic limit of SEQ, or any instrument with fixed Ei
    I = np.copy(seq_MD.getSignalArray())
    err = np.sqrt(np.copy(seq_MD.getErrorSquaredArray()))
    if type(Ei)==list:
        Q_arr,E_max = minQseq_multEi(Ei,twoThetaMin)
    else:
        Q_arr,E_max = minQseq(Ei,twoThetaMin)
    out_MD = seq_MD.clone()
    dims = seq_MD.getNonIntegratedDimensions()
    q_values = mdu.dim2array(dims[0])
    energies = mdu.dim2array(dims[1])
    for i in range(len(I)):
        q_cut = I[i]
        q_val = q_values[i]
        err_cut = err[i]
        kinematic_E = E_max[np.argmin(np.abs(q_val-Q_arr))]
        q_cut[np.where(energies>kinematic_E)]=np.nan
        err_cut[np.where(energies>kinematic_E)]=np.nan
        I[i]=q_cut
        err[i]=err_cut
    out_MD.setSignalArray(I)
    out_MD.setErrorSquaredArray(err**2)
    return out_MD

def mask_minQ_fixedEf_MD(seq_MD,twoThetaMin,Ef):
    #Remove areas outside of kinematic limit of MACS, or any instrument with fixed Ef
    I = np.copy(seq_MD.getSignalArray())
    err = np.sqrt(np.copy(seq_MD.getErrorSquaredArray()))
    if type(Ef)==list:
        Q_arr,E_max = minQmacs(Ef,twoThetaMin,deltaE=0)
    else:
        Q_arr,E_max =  minQmacs(Ef,twoThetaMin,deltaE=0)
    out_MD = seq_MD.clone()
    dims = seq_MD.getNonIntegratedDimensions()
    q_values = mdu.dim2array(dims[0])
    energies = mdu.dim2array(dims[1])
    Ef_arr = energies 
    for i in range(len(I)):
        q_cut = I[i]
        q_val = q_values[i]
        err_cut = err[i]
        kinematic_E = minQmacs(Ef,twoThetaMin,energies)[1]
        q_cut[np.where(energies>kinematic_E)[0]]=np.nan
        err_cut[np.where(energies>kinematic_E)[0]]=np.nan
        err_cut[q_cut==0]=np.nan
        q_cut[q_cut==0]=np.nan
        I[i]=q_cut
        err[i]=err_cut
    out_MD.setSignalArray(I)
    out_MD.setErrorSquaredArray(err**2)
    return out_MD