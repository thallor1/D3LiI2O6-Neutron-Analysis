import numpy as np
from mantidFF import get_MANTID_magFF
from progressbar import ProgressBar
from cut_mdhisto_powder import cut_MDHisto_powder
import os
import MDUtils as mdu
import lmfit
from lmfit import Model,Parameters
from splinefuncs import *
import matplotlib.pyplot as plt


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

def MDfactorization(workspace_MDHisto,mag_ion='Ir4',q_lim=False,e_lim=False,Ei=50.0,twoThetaMin=3.5,plot_result=True,method='powell',fname='placeholder.txt',\
                        fast_mode=False,overwrite_prev=False,allow_neg_E=True,g_factor=2.0):
    #Does a rocking curve foreach parameter to determine its uncertainty
    #Assumes that z is already flatted into a 1D array
    #This is specifically for the factorization technique
    #below contents are from old version of function 
    if overwrite_prev==True and os.path.exists(fname):
        #Delete the file
        os.remove(fname)
    if os.path.isfile(fname):
        err_array = np.genfromtxt(fname)
    dims = workspace_MDHisto.getNonIntegratedDimensions()
    q_values = mdu.dim2array(dims[0])
    energies = mdu.dim2array(dims[1])
    intensities = np.copy(workspace_MDHisto.getSignalArray())
    errors = np.sqrt(np.copy(workspace_MDHisto.getErrorSquaredArray()))
    events = np.copy(workspace_MDHisto.getNumEventsArray())
    intensities/=events 
    errors /= events 
    if q_lim!=False and e_lim!=False:
        qmin=q_lim[0]
        qmax=q_lim[1]
        emin=e_lim[0]
        emax=e_lim[1]
    elif q_lim==False and e_lim!=False:
        #e_lim has not been defined.
        qmin=np.min(q_values)
        qmax=np.max(q_values)
        emin=e_lim[0]
        emax=e_lim[1]
    elif e_lim!=False and q_lim==False:
        #q-lim hasn't been defined
        qmin=q_lim[0]
        qmax=q_lim[1]
        emin=np.min(energies)
        emax=np.max(energies)
    else:
        qmin=np.min(q_values)
        qmax=np.max(q_values)
        emin=np.min(energies)
        emax=np.max(energies)

    intensities = intensities[np.intersect1d(np.where(q_values>=qmin),np.where(q_values<=qmax))]
    intensities = intensities[:,np.intersect1d(np.where(energies>=emin),np.where(energies<=emax))]
    errors = errors[np.intersect1d(np.where(q_values>=qmin),np.where(q_values<=qmax))]
    errors = errors[:,np.intersect1d(np.where(energies>=emin),np.where(energies<=emax))]
    energies = energies[np.intersect1d(np.where(energies>=emin),np.where(energies<=emax))]
    q_values = q_values[np.intersect1d(np.where(q_values>=qmin),np.where(q_values<=qmax))]

    #Remove areas outside of kinematic limit of SEQ
    if twoThetaMin!=False and (type(Ei) != list):
        Q_arr,E_max = minQseq(Ei,twoThetaMin)
    elif type(Ei)!=list and Ei!=False:
        Q_arr,E_max = minQseq(Ei,3.5)
            
    if type(Ei)==list:
        Q_arr,E_max = minQseq_multEi(Ei,twoThetaMin)
    '''
    for i in range(len(intensities)):
        q_cut = intensities[i]
        q_val = q_values[i]
        err_cut = errors[i]
        kinematic_E = E_max[np.argmin(np.abs(q_val-Q_arr))]
        q_cut[np.where(energies>kinematic_E)]=np.nan
        err_cut[np.where(energies>kinematic_E)]=np.nan
        intensities[i]=q_cut
        errors[i]=err_cut
    '''


    x = q_values
    y = energies
    z = intensities
    #Take big cuts of the dataset to get a good guess
    q_res = np.abs(q_values[1]-q_values[0])
    e_res = np.abs(energies[1]-energies[0])
    q_vals_guess,q_cut_guess_dat,q_cut_guess_errs = cut_MDHisto_powder(workspace_MDHisto,'|Q|',[qmin,qmax,q_res],[emin,emax])
    e_vals_guess,e_cut_guess_dat,e_cut_guess_errs=cut_MDHisto_powder(workspace_MDHisto,'DeltaE',[emin,emax,e_res],[qmin,qmax])
    q_cut_guess = np.zeros(len(q_values))
    e_cut_guess = np.zeros(len(energies))
    for i in range(len(x)):
        #Put the cloest values of the cut into the guess:
        Q_i = np.argmin(np.abs(x[i]-q_vals_guess))
        q_cut_guess[i] = q_cut_guess_dat[Q_i]
        if np.isnan(q_cut_guess[i]):
            q_cut_guess[i]=np.nanmean(q_cut_guess_dat)
    for i in range(len(y)):
        E_i = np.argmin(np.abs(y[i]-e_vals_guess))
        e_cut_guess[i]=e_cut_guess_dat[E_i]
        if np.isnan(e_cut_guess[i]):
            e_cut_guess[i]=np.nanmean(e_cut_guess_dat)
    m = len(y) # number of E-values
    n = len(x) #number of q_values
    Q_cut = q_cut_guess.reshape(1,n)
    E_cut = e_cut_guess.reshape(m,1)
    xy=Q_cut*E_cut
    arr_guess = np.append(q_cut_guess,e_cut_guess)
    arr_guess[np.isnan(arr_guess)]=0
    params= Parameters()
    for i in range(len(arr_guess)):
        val = arr_guess[i]
        if i>n:
            #Spectral weight can't be negative physically
            if allow_neg_E==False:
                params.add('param_'+str(i),value=arr_guess[i],min=0)
            else:
                params.add('param_'+str(i),value=arr_guess[i])
        else:
            if allow_neg_E==False:
                params.add('param_'+str(i),value=arr_guess[i])
            else:
                params.add('param_'+str(i),value=arr_guess[i])


    weights = np.ones(np.shape(intensities))
    weights = 1.0/(np.abs(errors))

    #Make note of q, e indices at which there exist no intensities. They will be masked.
    bad_q_i=[]
    bad_e_i=[]
    for i in range(np.shape(intensities)[0]):
        #Check Q-cuts
        q_cut = intensities[i]
        num_nan = np.sum(np.isnan(q_cut))
        num_zero = np.sum([q_cut==0])
        num_bad = num_nan+num_zero
        if num_bad==len(q_cut):
            bad_q_i.append(i)
        else:
            #Do nothing
            pass
    #Some high energies will also have no counts from detailed balance
    for i in range(np.shape(intensities)[1]):
        e_cut = intensities[:,i]
        num_nan = np.sum(np.isnan(e_cut))
        num_zero = np.sum([e_cut==0])
        num_bad =num_nan+num_zero

        if num_bad==len(e_cut):
            bad_e_i.append(i)
        else:
            #Do nothing
            pass
    weights=np.ravel(weights)
    meas_errs=1.0 / weights 
    z_fit = np.copy(intensities)
    Q,E = np.meshgrid(e_vals_guess,q_vals_guess)
    z_fit_orig = np.copy(z_fit)
    z_fit = z_fit.flatten()
    #weights[z_fit==0]=0

    z_fit[z_fit==0]=np.nan
    meas_errs[np.isnan(z_fit)]=np.nan

    xy = np.arange(z_fit.size)
    num_points = len(z_fit) - np.sum(np.isnan(z_fit)) - np.sum([z_fit==0])
    vals = []
    for i in range(len(params)):
        vals.append(params['param_'+str(i)].value)
    vals = np.array(vals)
    Q_vals = vals[0:n].reshape(n,1) #Reconstruct y coordinates
    E_vals = vals[n:].reshape(1,m) # '' x coord
    slice2D = Q_vals * E_vals

    def f(params):
        vals = []
        for i in range(len(params)):
            vals.append(params['param_'+str(i)])
        vals = np.array(vals)
        Q_vals = vals[0:n].reshape(n,1) #Reconstruct y coordinates
        E_vals = vals[n:].reshape(1,m) # '' x coord

        slice2D = Q_vals * E_vals
        
        slice2D[np.isnan(z_fit_orig)]=np.nan
        Q,E = np.meshgrid(q_vals_guess,e_vals_guess)
        obs_I = z_fit.reshape((n,m))
        calcI= slice2D.flatten()
        chisqr = np.nansum((z_fit - calcI)**2 / (meas_errs**2))/num_points
        return chisqr
    chisqr0 = f(params)
    #minimize this chisqr function. 
    result = lmfit.minimize(fcn=f,params=params,method='powell')
    chisqr=result.residual
    chisqr0 = f(result.params)

    denom = (1.0+1.0/(float(len(z_fit))-float(len(result.params))))
    chisqrmin=chisqr0/(1.0+1.0/(float(num_points)-float(len(result.params))))
    chisqrmax=chisqr0/(1.0-1.0/(float(num_points)-float(len(result.params))))

    f_array=[]
    for i in range(len(result.params)):
        f_array.append(result.params['param_'+str(i)].value)
    num_operations=len(params)
    #Normalize s.t. energy spectra integrates to one
    q = q_values
    e = energies
    x_q = np.array(f_array[0:n])
    x_q[bad_q_i]=0
    g_e = np.array(f_array[n:])
    if os.path.isfile(fname) or fast_mode==True:
        if fast_mode==True:
            err_array =0.3*np.array(f_array)
        x_q_errs = err_array[0:n]
        g_e_errs = err_array[n:]
        #Need to normalize

        #Normalize s.t. energy spectra integrates to one
        q = q_values
        e = energies
        x_q = np.array(f_array[0:n])
        x_q[bad_q_i]=0
        g_e = np.array(f_array[n:])
        g_e[bad_e_i]=0
        g_e[np.isnan(g_e)]=0
        x_q[np.isnan(x_q)]=0

        ge_int = np.trapz(g_e,x=energies*1.0) #This integral needs to be in eV, not meV
        xq_err = err_array[0:n]
        ge_err = err_array[n:]

        x_q = np.array(x_q)*ge_int
        xq_err = np.array(xq_err)*ge_int
        g_e = np.array(g_e)/ge_int
        ge_err = np.array(ge_err)/ge_int

        #Now convert X(Q) into S(Q)
        r0 = 0.5391
        g=g_factor
        q_FF,magFF = get_MANTID_magFF(q,mag_ion)
        magFFsqr = 1.0/np.array(magFF)
        s_q = (2.0*x_q)/(r0**2 * g**2 * magFFsqr)
        s_q_err = (2.0*xq_err)/(r0**2 * g**2 * magFFsqr)
        return q,s_q,s_q_err,e,g_e,ge_err
    #Make splines to try to estimate deviation from mean factorization
    # smooth(x,window_len=11,window='hanning')
    Nx=len(q)/10
    Ny = len(e)/10

    #xq_padded = np.pad(x_q, (Nx//2, Nx-1-Nx//2), mode='edge')
    #x_smooth = np.convolve(xq_padded, np.ones((Nx,))/Nx, mode='valid') 
    #ge_padded = np.pad(g_e, (Ny//2, Ny-1-Ny//2), mode='edge')
    #g_smooth = np.convolve(ge_padded, np.ones((Ny,))/Ny, mode='valid')
    try:
        model_xq_smooth = get_natural_cubic_spline_model(q,x_q,minval=np.nanmin(q),maxval=np.nanmax(q),n_knots=round(Nx)+1)
        x_smooth = model_xq_smooth.predict(q)
        model_ge_smooth = get_natural_cubic_spline_model(e,g_e,minval=np.nanmin(e),maxval=np.nanmax(e),n_knots=round(Ny)+1)
        g_smooth=model_ge_smooth.predict(e)
        x_diff = np.abs(x_q - x_smooth)
        g_diff = np.abs(g_e - g_smooth)
        x_q_stdDev = np.mean(x_diff)
        g_e_stdDev = np.mean(g_diff)
        if Ny<2:
            g_e_stdDev = np.mean(g_e)/10.0
        if Nx<2:
            x_q_stdDev=np.mean(x_q)/10.0

    except Exception as e:
        g_e_stdDev = np.mean(g_e)/10.0
        x_q_stdDev=np.mean(x_q)/10.0


    errs={}
    err_array=[]
    count=0
    show_plots=True
    extrapolate=True
    pre_fit = False
    num_operations = len(result.params)
    #Get spacing between parameters to determine 
    param_val_list = []
    for param in result.params:
        value = result.params[param].value 
        param_val_list.append(value)
    std_dev_params = np.std(np.array(param_val_list))
    progress = ProgressBar(num_operations, fmt=ProgressBar.FULL)
    prev_slope=False
    for i in range(len(result.params)):
        fitnow=False
        paramkey = list(result.params.keys())[i]
        param = result.params[paramkey]
        mean_val = param.value
        test_value_arr=[]
        chisqr_arr=[]
        param_vals = []
        if i>n:
            stdDev = g_e_stdDev
        else:
            stdDev = x_q_stdDev
        min_param_val = param.value-stdDev*3.0
        max_param_val = param.value+stdDev*3.0
        step = (max_param_val - min_param_val)/100.0
        j=0
        l=0
        flag=0
        flag1=False
        flag2=False
        while flag1==False or flag2==False:
            if j%2==0:
                #On an even run- seek out next positive value
                new_val = mean_val+l*step
                if new_val<=max_param_val:
                    param_vals.append(new_val)
                else:
                    flag1=True
            else:
                #Odd now- seek out negative value.
                new_val = mean_val-l*step
                if new_val>=min_param_val:
                    param_vals.append(new_val)
                else:
                    flag2=True
                #increase the counter - both + and - sides have been addressed
                l=l+1
            j=j+1
        new_params = result.params.copy()
        opt_val = mean_val
        prev_chisqr_pos = chisqr0*0.0
        prev_chisqr_neg = 0.0
        for val in param_vals:
            new_params.add(paramkey,vary=False,value=val)
            #Determine side of parabola
            if (val-opt_val)>0:
                side='pos'
            else:
                side='neg'
            #Use the side of the parabola to compare to the closest chisqr
            if side=='pos':
                oldchisqr = prev_chisqr_pos
            else:
                oldchisqr = prev_chisqr_neg
            #Determine a new chisqr for this point on the parabols

            new_result = lmfit.minimize(fcn=f,params=new_params,method='powell')
            new_chisqr=f(new_result.params)
            #If this new chisqr is greater than the maximum allowed chisqr, then we've found our error.
            #Call this case A
            if new_chisqr>chisqrmax and len(chisqr_arr)>=7:
                error=np.abs(float(opt_val)-val)
                fitnow=True
                flag=1
            if new_chisqr<oldchisqr and len(chisqr_arr)>=7:
                #local minima shifted, no longer good.
                fitnow=True

            #If the new chisqr is greater than the previous one all is well. 
            #Append the new chisqr and test value to the array. 
            if new_chisqr>oldchisqr or len(chisqr_arr)<7:
                chisqr_arr.append(new_chisqr)
                test_value_arr.append(val)
                if side=='pos':
                    prev_chisqr_pos=new_chisqr
                else:
                    prev_chisqr_neg=new_chisqr
            #Perform parabola extrapolation at this point if fitnow flag is active
            if (len(chisqr_arr)>11 and extrapolate==True) or (fitnow==True and len(chisqr_arr)>=7):
                #We've tested enough- fit this to a parabola
                flag = 1
                def parabola(x,a,b,c):
                    return a*((x-b)**2)+c
                para_model = Model(parabola)
                para_params = para_model.make_params()
                #very roughly assume it's linear between 0 and 1 points
                if prev_slope==False:
                    guess_slope=np.abs(chisqrmax-chisqr0)/(((np.max(np.array(test_value_arr))-opt_val)**2))
                else:
                    guess_slope = prev_slope
                #Handle the case of b=0
                if np.abs(opt_val)<1e-5:
                    bmin=-0.05
                    bmax=0.05
                else:
                    bmin=0.9*opt_val-1e-5
                    bmax=1.1*opt_val+1e-5
                para_params.add('a',value=guess_slope*1.0)
                para_params.add('b',vary=False,value=opt_val,min=bmin,max=bmax)
                para_params.add('c',vary=False,value=np.min(chisqr_arr))
                para_fit = para_model.fit(chisqr_arr,x=test_value_arr,params=para_params,method='powell')
                a_fit = para_fit.params['a'].value
                b_fit = para_fit.params['b'].value
                c_fit = para_fit.params['c'].value
                prev_slope = a_fit
                print('fit A'+str(a_fit))
                print('fit B'+str(b_fit))
                print('fit C'+str(c_fit))
                max_param_val = np.sqrt(np.abs((chisqrmax-c_fit)/a_fit))+b_fit
                #print('Max param_val='+str(max_param_val))
                error = np.abs(max_param_val-b_fit)
                err_array.append(error)
                opt_val=b_fit
                eval_range = np.linspace(opt_val-error*1.2,opt_val+error*1.2,3000)
                fit_eval = para_model.eval(x=eval_range,params=para_fit.params)
                errs[param.name]=error
                flag=1
                if len(param_vals)==0:
                    param_vals=[opt_val]
                if show_plots==True and len(param_vals)>1:
                    try:
                        plt.figure()
                        plt.plot(test_value_arr,chisqr_arr,color='k',marker='o',ls='')
                        plt.xlabel('Param val')
                        plt.ylabel('Chisqr')
                        plt.title('Error ='+str(round(error,3))+' or '+str(round(100*error/opt_val,3))+'%')
                        plt.xlim(np.min(test_value_arr)-np.abs(np.min(test_value_arr))/10.0,np.max(test_value_arr)*1.1)
                        plt.plot(np.linspace(0,np.max(param_vals)+1e-9,10),np.ones(10)*np.abs(chisqrmax),'r')
                        plt.plot(eval_range,fit_eval,'b--')
                        plt.plot(opt_val+error,chisqrmax,'g^')
                        plt.plot(opt_val-error,chisqrmax,'g^')
                        plt.ylim(np.min(chisqr_arr),np.abs(chisqrmax)+(chisqrmax-chisqrmin)/5.0)
                        plt.xlim(0.9*np.min(test_value_arr),np.max(test_value_arr)*1.1)
                        plt.show()

                    except Exception as e:
                        print('Some error while plotting.')
                        print(e)
                    #plt.ylim(0.0,1.3*np.abs(chisqrmax))

                break
        progress.current+=1
        pre_fit=False
        progress()
    if not(os.path.isfile(fname)):
        np.savetxt(fname,err_array)
    f_array=[]
    for i in range(len(result.params)):
        f_array.append(result.params['param_'+str(i)].value)
    num_operations=len(params)
    #Normalize s.t. energy spectra integrates to one
    q = q_values
    e = energies
    x_q = np.array(f_array[0:n])
    x_q[bad_q_i]=0
    g_e = np.array(f_array[n:])
    g_e[bad_e_i]=0
    g_e[np.isnan(g_e)]=0
    x_q[np.isnan(x_q)]=0

    ge_int = np.trapz(g_e,x=energies)
    xq_err = err_array[0:n]
    ge_err = err_array[n:]

    x_q = np.array(x_q)*ge_int
    xq_err = np.array(xq_err)*ge_int
    g_e = np.array(g_e)/ge_int
    ge_err = np.array(ge_err)/ge_int

    #Now convert X(Q) into S(Q)
    r0 = 0.5391
    g=g_factor
    q_FF,magFF = get_MANTID_magFF(q,mag_ion)
    magFFsqr = 1.0/np.array(magFF)
    s_q = (2.0*x_q)/(r0**2 * g**2 * magFFsqr)
    s_q_err = (2.0*xq_err)/(r0**2 * g**2 * magFFsqr)
    return q,s_q,s_q_err,e,g_e,ge_err

def calc_chisqr_val(obs_arr,theory_arr,obs_err_arr,theory_err_arr=0.0):
    #For arbitrary arrays of theory, experiment, errors, returns the chisqr statistic.
    #Returns it for each point I guess. 
    N = np.nansum([obs_err_arr<1e9])
    chisqr = 0
    obs_arr = np.array(obs_arr)
    theory_arr = np.array(theory_arr)
    obs_err_arr = np.array(obs_err_arr)
    diffsqr = (obs_arr - theory_arr)**2
    chisqr_arr = diffsqr / obs_err_arr**2
    chisqr = np.nansum(chisqr_arr) / N
    return chisqr

def factorization_f(n,m,delE,**vals):
    #Returns flattened calculated spectra from factorization
    vals_arr = []
    for i in range(len(vals)):
        vals_arr.append(vals['param_'+str(i)])
    vals = np.array(vals_arr)
    Q_vals = vals[0:n].reshape(n,1) #Reconstruct y coordinates
    #Update to this function means that these are delta vals in e^(-delta_i)/Z
    deltas = vals[n:]
    Z = np.nansum(np.exp(-1.0*deltas))
    E_vals = np.exp(-1.0*deltas)/(delE*Z) 

    E_vals = E_vals.reshape(1,m) # '' x coord

    slice2D = Q_vals * E_vals

    calcI = slice2D.flatten()
    #chisqr = np.nansum((z_fit - calcI)**2 / (meas_errs**2))/num_points
    return calcI


def MDfactorizationv2(workspace_MDHisto,mag_ion='Ir4',q_lim=False,e_lim=False,Ei=50.0,twoThetaMin=3.5,plot_result=True,method='powell',fname='placeholder.jpg',\
                        fast_mode=False,overwrite_prev=False,allow_neg_E=True,g_factor=2.0,debug=False,fix_Qcut=False,fix_Ecut=False):
    #Does a rocking curve foreach parameter to determine its uncertainty
    #Assumes that z is already flatted into a 1D array
    #This is specifically for the factorization technique
    #below contents are from old version of function 

    #Version 2 includes an updated definition of F(omega)
    if overwrite_prev==True and os.path.exists(fname):
        #Delete the file
        os.remove(fname)

    dims = workspace_MDHisto.getNonIntegratedDimensions()
    q_values = mdu.dim2array(dims[0])
    energies = mdu.dim2array(dims[1])
    intensities = np.copy(workspace_MDHisto.getSignalArray())
    errors = np.sqrt(np.copy(workspace_MDHisto.getErrorSquaredArray()))
    if q_lim!=False and e_lim!=False:
        qmin=q_lim[0]
        qmax=q_lim[1]
        emin=e_lim[0]
        emax=e_lim[1]
    elif q_lim==False and e_lim!=False:
        #e_lim has not been defined.
        qmin=np.min(q_values)
        qmax=np.max(q_values)
        emin=e_lim[0]
        emax=e_lim[1]
    elif e_lim!=False and q_lim==False:
        #q-lim hasn't been defined
        qmin=q_lim[0]
        qmax=q_lim[1]
        emin=np.min(energies)
        emax=np.max(energies)
    else:
        qmin=np.min(q_values)
        qmax=np.max(q_values)
        emin=np.min(energies)
        emax=np.max(energies)

    intensities = intensities[np.intersect1d(np.where(q_values>=qmin),np.where(q_values<=qmax))]
    intensities = intensities[:,np.intersect1d(np.where(energies>=emin),np.where(energies<=emax))]
    errors = errors[np.intersect1d(np.where(q_values>=qmin),np.where(q_values<=qmax))]
    errors = errors[:,np.intersect1d(np.where(energies>=emin),np.where(energies<=emax))]
    energies = energies[np.intersect1d(np.where(energies>=emin),np.where(energies<=emax))]
    q_values = q_values[np.intersect1d(np.where(q_values>=qmin),np.where(q_values<=qmax))]

    #Remove areas outside of kinematic limit of SEQ 
    #This is outside of the scope of the script.
    if twoThetaMin!=False and Ei!=False:
        if twoThetaMin!=False and (type(Ei) != list):
            Q_arr,E_max = minQseq(Ei,twoThetaMin)
        elif type(Ei)!=list:
            Q_arr,E_max = minQseq(Ei,twoThetaMin)
                
        if type(Ei)==list:
            Q_arr,E_max = minQseq_multEi(Ei,twoThetaMin)
        for i in range(len(intensities)):
            q_cut = intensities[i]
            q_val = q_values[i]
            err_cut = errors[i]
            kinematic_E = E_max[np.argmin(np.abs(q_val-Q_arr))]
            q_cut[np.where(energies>kinematic_E)]=np.nan
            err_cut[np.where(energies>kinematic_E)]=np.nan
            intensities[i]=q_cut
            errors[i]=err_cut
    


    x = q_values
    y = energies
    z = intensities
    bad_i = np.isnan(z)
    intensities[np.isnan(z)]=0
    errors[np.isnan(z)]=1e10
    errors[errors==0]=1e10
    errors[np.isnan(errors)]=1e10

    #Take big cuts of the dataset to get a good guess
    q_res = np.abs(q_values[1]-q_values[0])
    e_res = np.abs(energies[1]-energies[0])
    
    q_vals_guess = np.copy(x)
    q_cut_guess = np.zeros(len(x))
    q_cut_guess_errs=np.zeros(len(x))
    e_cut_guess = np.zeros(len(y))
    e_cut_guess_errs = np.zeros(len(y))
    for i in range(len(q_cut_guess)):
        q_i = intensities[i,:]
        qerr_i = errors[i,:]
        qpt = np.average(q_i,weights=1.0/qerr_i)
        q_cut_guess[i]=qpt 
        q_cut_guess_errs[i]=qpt*np.mean(q_i/qerr_i)
    for i in range(len(e_cut_guess)):
        e_i = intensities[:,i]
        eerr_i = errors[:,i]
        ept=np.average(e_i,weights=1.0/eerr_i)
        e_cut_guess[i]=ept 
        e_cut_guess_errs[i]=ept*np.mean(e_i/eerr_i)
    
    #q_vals_guess,q_cut_guess,q_cut_guess_errs = cut_MDHisto_powder(workspace_MDHisto,'|Q|',[np.nanmin(x)-q_res/2.0,np.nanmax(x)+q_res/2.0,q_res/10.0],e_lim)
    #e_vals_guess,e_cut_guess,e_cut_guess_errs = cut_MDHisto_powder(workspace_MDHisto,'DeltaE',[np.nanmin(y)-e_res/2.0,np.nanmax(y)+e_res/2.0,e_res/10.0],q_lim)

    e_vals_guess = y 
    q_vals_guess = x
    #Normalize the cuts to one in energy
    e_cut_guess[np.where(e_cut_guess<=0)[0]]=1e-2
    e_cut_integral = np.trapz(x=y,y=e_cut_guess)
    print('Gw integral')
    print(e_cut_integral)
    e_cut_guess/=e_cut_integral
    q_cut_guess*=e_cut_integral
    #Convert the actual cut into deltas used in the expoential definition of G(omega)
    #We do this by defining the delta at the first value in the cut to be zero. 
    g_omega_0 = e_cut_guess[0]
    delta_0 = 0.0
    Z = 1.0/g_omega_0 #This is used to solve for delta now. 
    delta_arr = np.zeros(len(e_cut_guess))
    delta_arr = np.log(1.0/e_cut_guess*Z)
    calc_ecut_guess = np.exp(-1.0*delta_arr)/Z

    m = len(y) # number of E-values
    n = len(x) #number of q_values
    #e_cut_guess/=e_cut_guess[0]
    #q_cut_guess*=e_cut_guess[0]
    Q_cut = q_cut_guess.reshape(1,n)
    E_cut = e_cut_guess.reshape(m,1)
    xy=Q_cut*E_cut
    arr_guess = np.append(q_cut_guess,delta_arr)
    arr_guess[np.isnan(arr_guess)]=0
    params= Parameters()
    for i in range(len(arr_guess)):
        val = arr_guess[i]
        if i>=n:
            #Spectral weight can't be negative physically
            #Need to fix the first energy value 
            if i==n:
                vary_val = False
                param_guess = 0.0
            else:
                if fix_Ecut==True:
                    vary_val=False
                else:
                    vary_val=True
                param_guess = arr_guess[i]
            #From of G(w) is e^(-delta), e^-15 is 1e-7
            params.add('param_'+str(i),vary=vary_val,value=param_guess,max=15.0)
        else:
            if fix_Qcut == True:
                vary_val=False
            else:
                vary_val=True
            params.add('param_'+str(i),value=val,vary=vary_val)
    if debug==True:
        plt.figure()
        plt.errorbar(x,q_cut_guess,q_cut_guess_errs,color='k')
        plt.show()
        plt.figure()
        plt.errorbar(y,calc_ecut_guess,e_cut_guess_errs,color='k')
        plt.errorbar(y,e_cut_guess,e_cut_guess_errs,color='r')
        plt.show()

    weights = np.ones(np.shape(intensities))
    weights = 1.0/(np.abs(errors))

    #Make note of q, e indices at which there exist no intensities. They will be masked.
    bad_q_i=[]
    bad_e_i=[]
    for i in range(np.shape(intensities)[0]):
        #Check Q-cuts
        q_cut = intensities[i]
        num_nan = np.sum(np.isnan(q_cut))
        num_zero = np.sum([q_cut==0])
        num_bad = num_nan+num_zero
        if num_bad==len(q_cut):
            bad_q_i.append(i)
        else:
            #Do nothing
            pass
    #Some high energies will also have no counts from detailed balance
    for i in range(np.shape(intensities)[1]):
        e_cut = intensities[:,i]
        num_nan = np.sum(np.isnan(e_cut))
        num_zero = np.sum([e_cut==0])
        num_bad =num_nan+num_zero
        if num_bad==len(e_cut):
            bad_e_i.append(i)
        else:
            #Do nothing
            pass
    weights=np.ravel(weights)
    meas_errs=1.0 / weights 

    z_fit = np.copy(intensities)
    Q,E = np.meshgrid(e_vals_guess,q_vals_guess)
    z_fit_orig = np.copy(z_fit)
    z_fit = z_fit.flatten()
    #weights[z_fit==0]=0
    meas_errs[np.isnan(z_fit)]=np.inf

    z_fit[np.isnan(z_fit)]=np.nan

    xy = np.arange(z_fit.size)
    num_points = len(z_fit) - np.sum(np.isnan(z_fit)) - np.sum([z_fit==0])
    vals = []
    for i in range(len(params)):
        vals.append(params['param_'+str(i)].value)
    vals = np.array(vals)
    Q_vals = vals[0:n].reshape(n,1) #Reconstruct y coordinates
    E_vals = vals[n:].reshape(1,m) # '' x coord
    slice2D = Q_vals * E_vals


    weights = 1.0/meas_errs
    data = z_fit 
    data[np.isnan(data)]=0
    weights[data==0]=0
    meas_errs=1.0/weights 
    meas_errs[weights==0]=np.nan
    model = Model(factorization_f,independent_vars=['n','m','delE'])
    eRes = np.abs(energies[1]-energies[0])
    #minimize this chisqr function. 
    result = model.fit(data,n=n,m=m,delE=eRes,params=params,method=method,weights=weights,nan_policy='omit')
    f_array=[]
    for i in range(len(result.params)):
        f_array.append(result.params['param_'+str(i)].value)


    num_operations=len(params)
    #Normalize s.t. energy spectra integrates to one
    q = q_values
    e = energies
    x_q = np.array(f_array[0:n])
    x_q[bad_q_i]=0
    deltas = np.array(f_array[n:])
    Z = np.nansum(np.exp(-1.0*deltas))
    g_e = np.exp(-1.0*deltas)/(eRes*Z) 
    err_dict={}
    if os.path.isfile(fname) or fast_mode==True:
        if fast_mode==True:
            err_array =0.3*np.array(f_array)
        else:
            err_dict = np.load(fname,allow_pickle=True).item()
            err_array=[]
            for i in range(len(err_dict.keys())):
                key = 'param_'+str(i)
                err_array.append(err_dict[key])
        x_q_errs = err_array[0:n]
        delta_errs = err_array[n:]
        #Need to normalize

        #Normalize s.t. energy spectra integrates to one
        q = q_values
        e = energies
        x_q = np.array(f_array[0:n])
        x_q[bad_q_i]=0
        deltas = np.array(f_array[n:])
        Z = np.nansum(np.exp(-1.0*deltas))
        g_e = np.exp(-1.0*deltas)/(eRes*Z) 

        xq_err = err_array[0:n]
        ge_err = err_array[n:]

        x_q,g_e,xq_err,ge_err = np.array(x_q),np.array(g_e),np.array(xq_err),np.array(ge_err)

        #Now convert X(Q) into S(Q)
        r0 = 0.5391
        g=g_factor
        q_FF,magFF = get_MANTID_magFF(q,mag_ion)
        magFFsqr = 1.0/np.array(magFF)
        s_q = (2.0*x_q)/(r0**2 * g**2 * magFFsqr)
        s_q_err = (2.0*xq_err)/(r0**2 * g**2 * magFFsqr)
        return q,s_q,s_q_err,e,g_e,ge_err
    #Get error bars using random sampling method
    #Create a parameter object mased on linearized form of the factorization for uncertainty
    if len(err_dict.keys())!=len(f_array):
        #If it is the same, the dictionary of the appropriate dimension has already been loaded.
        err_dict = calculate_param_uncertainty(data,meas_errs,model,result.params,result,fast_calc=fast_mode,independent_vars={'n':n,'m':m,'delE':eRes},\
            extrapolate=False,show_plots=True,fname=fname,overwrite_prev=overwrite_prev,num_test_points = 20,debug=False,fit_method=method)
    err_array = []
    for i in range(len(result.params)):
        f_array.append(result.params['param_'+str(i)].value)
        err_array.append(err_dict['param_'+str(i)])
    err_array=np.array(err_array)
    num_operations=len(params)
    #Translate delta values to g_e
    #Z = np.nansum(np.exp(-1.0*np.array(f_array[n:])))
    #g_e = np.exp(-1.0*np.array(f_array[n:]))
    q = q_values
    e = energies
    x_q = np.array(f_array[0:n])
    x_q[bad_q_i]=0
    g_e[bad_e_i]=0
    g_e[np.isnan(g_e)]=0
    x_q[np.isnan(x_q)]=0
    ge_int = np.trapz(g_e,x=energies)
    xq_err = err_array[0:n]
    delta_err = np.array(err_array[n:])
    ge_err = g_e*deltas*delta_err#propogate the error

    x_q = np.array(x_q)
    xq_err = np.array(xq_err)
    g_e = np.array(g_e)
    ge_err = np.array(ge_err)
    #Now convert X(Q) into S(Q)
    r0 = 0.5391
    g=g_factor
    q_FF,magFF = get_MANTID_magFF(q,mag_ion)
    magFFsqr = 1.0/np.array(magFF)
    s_q = (2.0*x_q)/(r0**2 * g**2 * magFFsqr)
    s_q_err = (2.0*xq_err)/(r0**2 * g**2 * magFFsqr)
    if plot_result==True:
        try:
            plt.figure()
            plt.errorbar(q,s_q*magFFsqr,yerr=s_q_err*magFFsqr,capsize=3,ls=' ',mfc='w',mec='k',color='k',marker='',label=r'S(Q)|M(Q)|$^2$')
            plt.errorbar(q,s_q,yerr=xq_err,capsize=3,ls=' ',mfc='w',mec='b',color='b',marker='',label=r'S(Q)')
            plt.legend()
            plt.title('S(Q) Factorization Result')
            plt.xlabel('Q($\AA^{-1}$))')
            plt.ylabel('S(Q) barn/mol/f.u.')
            plt.figure()
            plt.title('G($\omega$) Factorization Result')
            plt.xlabel('$\hbar$\omega (meV)')
            plt.ylabel('G($\omega$) (1/meV)')
            plt.errorbar(energies,g_e,yerr=ge_err,capsize=3,ls=' ',mfc='w',mec='k',color='k',marker='')
        except Exception as e:
            print('Error when trying to plot result:')
            print(e)
    #Finally, save result

    return q,s_q,s_q_err,e,g_e,ge_err



def calculate_param_uncertainty(obs_vals,obs_errs,model,params,result,independent_vars=False,fast_calc=False,extrapolate=False,show_plots=True,fname='test.jpg',overwrite_prev=False,num_test_points = 30,debug=False,fit_method='powell'):
    '''
    This is a function to calculate the uncertainties in the free parameters of any lmfit model.
    It assumes a parabolic form, and takes the following arguments:
        obs_vals- np array - experimental values of the function
        obs_errs- np array - experimental errors of the function
        model - lmfit model describing the fitting function
        params - lmfit parameter object
        result - lmfit results of the best fit for the function
        indpendent_vars- bool or dict - if the model requires indepenedent vars, they are included here in the form a dictionary (i.e. independent_vars={'x':x,'y',y} in the function call)
        fast_calc - bool - spits out values if just testing and don't need to evaluate perfectly.
        extrapolate -bool- assumes a parabolic form for chisqr and gets uncertainties based around that. 
        show_plots - bool - shows the parabolic fits or the raw calculations
        fname - string- filename to store results 
        overwrite_prev- bool - determines if previous results should be loaded from file or overwritten. 

    Returns a dictionary of param names and errors
    '''
    #First step is to check if the parameters already exist.
    if os.path.isfile(fname) and overwrite_prev==True:
        os.remove(fname)
    if os.path.isfile(fname) and overwrite_prev!=True:
        errors=np.load(fname,allow_pickle=True).item()
        return errors
    if fast_calc==True:
        errs={}
        for param in result.params:
            errs[param]=result.params[param].value*0.2
        return errs 
    #Get an initial value of chisqr
    obs_errs[obs_errs==0]=np.nan 
    obs_errs[np.isnan(obs_errs)]=np.inf
    obs_vals[obs_vals==0]=0
    obs_vals[np.isnan(obs_vals)]=0
    chisqr0=calc_chisqr_val(obs_vals,result.best_fit,obs_errs)
    #Get the number of free parameters in the fit:
    num_free_params=0
    for param in result.params:
        if result.params[param].vary==True:
            num_free_params+=1
    #Get the number of points being fit
    num_points = np.nansum([obs_errs<1e10])
    #Calculate the statistical min and max allowed values of chisqr
    chisqrmin=chisqr0/(1.0+1.0/(num_points-num_free_params))
    chisqrmax=chisqr0/(1.0-1.0/(num_points-num_free_params))
    if debug==True:
        print('Chisqr0='+str(chisqr0))
        print('Chisqrmax='+str(chisqrmax))
    #Calculated errors will be returned in this dictionary:
    err_out = {}
    #Evaluated points will be kept track of in a param val list and a chisqr list

    progress = ProgressBar(num_free_params, fmt=ProgressBar.FULL) #I like a progress bar
    #Now we run into the problem of which points to test. This needs to be an adaptive process. 
    '''
    General algorithm for which points to test is the following:
    1. Set a min/max param value based on a percent error, noting that near zero a pure percent error will fail. 
    2. Test each of these points. If they are above the max value, then reduce the percent error by a factor of two. 
            If they are below the max value, then increase the percent error by a factor of two. 
            This should result in wide adaptability for ranges of fit parameters. 
    3. Once the two border points have been found (the points directly above and below the max chisqr)
        see if the number of evaluated points is acceptable. If not, fill in the parameter space between the points 
        uniformly to find an acceptable number of points. 
    4. Once a suitable number of points have been evaluated, perform a parabolic fit to determine the error bar if desired.
    5. If the parabolic fit is not performed, then the error bar is taken as half the distance between the min/max points.

    '''
    #Iterate through each parameter to do this 
    weights = 1.0 / obs_errs
    prev_slope = False
    for param in result.params:
        affect_chisqr=True
        try:
            if result.params[param].vary==True:
                print('Evaluating Uncertainty for '+str(param))
                found_min = False 
                found_max = False
                min_i = 0.0
                max_i = 0.0
                #Evaluated points will be kept track of in a param val list and a chisqr list
                param_list=[]
                chisqr_list = []
                init_param_val = result.params[param].value
                if debug==True:
                    print('Init param val')
                    print(init_param_val)
                #param_list.append(init_param_val)
                #chisqr_list.append(chisqr0)
                opt_val = init_param_val
                new_min_chisqr_prev = 0.0
                while found_min==False and min_i<1e2:
                    new_params_min=result.params.copy()
                    min_param_val = init_param_val - np.abs(init_param_val*(2.0**min_i)*0.005) #Start with a small 0.5% error 
                    new_params_min.add(param,vary=False,value=min_param_val)
                    if type(independent_vars)==bool:
                        new_result_min=model.fit(obs_vals,params=new_params_min,nan_policy='omit',method='powell',weights=weights)
                    else:
                        new_result_min=model.fit(obs_vals,params=new_params_min,nan_policy='omit',method='powell',weights=weights,**independent_vars)
                    #Get the chisqr after fitting the new param
                    new_min_chisqr = calc_chisqr_val(obs_vals,new_result_min.best_fit,obs_errs)
                    if new_min_chisqr>chisqrmax:
                        found_min=True #We are free
                    if new_min_chisqr<new_min_chisqr_prev:
                        #Jumped out of a local minima, need to break. Do not append these points to the array
                        found_min=True
                    else:
                        param_list.append(min_param_val)
                        chisqr_list.append(new_min_chisqr)

                        min_i = min_i+1.0
                        if debug==True: 
                            print(min_i)
                            print('Curr chisqr: '+str(new_min_chisqr))
                            print('Max chisqr: '+str(chisqrmax))
                        if new_min_chisqr==new_min_chisqr_prev and min_i>4:
                            print('Param '+str(param)+' does not affect chisqr.')
                            found_min=True
                            affect_chisqr=False
                        new_min_chisqr_prev=new_min_chisqr
                new_max_chisqr_prev=0.0
                while found_max==False and max_i<1e2:
                    new_params_min=result.params.copy()
                    max_param_val = init_param_val + np.abs(init_param_val*(2.0**max_i)*0.005)
                    new_params_max=result.params.copy()
                    new_params_max.add(param,vary=False,value=max_param_val)
                    if type(independent_vars)==bool:
                        new_result_max=model.fit(obs_vals,params=new_params_max,nan_policy='omit',method='powell',weights=weights)
                    else:
                        new_result_max=model.fit(obs_vals,params=new_params_max,nan_policy='omit',method='powell',weights=weights,**independent_vars)            
                    #Get the chisqr after fitting the new param
                    new_max_chisqr = calc_chisqr_val(obs_vals,new_result_max.best_fit,obs_errs)
                    if new_max_chisqr>chisqrmax:
                        found_max=True 
                    if new_max_chisqr<new_max_chisqr_prev:
                        #Jumped out of local minima, function is not well behaved.
                        found_max=True
                    else:
                        param_list.append(max_param_val)
                        chisqr_list.append(new_max_chisqr)

                        if new_max_chisqr==new_max_chisqr_prev and max_i>4:
                            print('Param '+str(param)+' does not affect chisqr.')
                            affect_chisqr=False
                            found_max=True
                        max_i = max_i+1
                        if debug==True: 
                            print(max_i)
                            print('Curr chisqr: '+str(new_max_chisqr))
                            print('Max chisqr: '+str(chisqrmax))
                        new_max_chisqr_prev=new_max_chisqr
                if found_min==False or found_max==False:
                    print('WARNING- strange behavior in uncertainty calculation, enbable show_plots==True to assure correctness')
                #Supposedly they have both been found. Check the number of points. if the estimate was initially way too large, then we need to fill them in . 
                num_eval_points = np.sum([np.array(chisqr_list)<chisqrmax])
                max_point = np.max(param_list)
                min_point = np.min(param_list)
                if debug==True:
                    print('Max Point: ')
                    print(max_point)
                    print('Min point: ')
                    print(min_point)
                    print('Num eval points:')
                    print(num_eval_points)
                while num_eval_points<num_test_points:
                    #Evaluate the remainder of points in an evenly spaced fashion
                    num_new_points = int(1.5*(num_test_points-num_eval_points))
                    #fill_points = np.linspace(min_point,max_point,num_new_points)
                    fill_points = np.random.uniform(low=min_point,high=max_point,size=num_new_points)
                    for param_val in fill_points:
                        new_params=result.params.copy()
                        new_params.add(param,vary=False,value=param_val)
                        if type(independent_vars)==bool:
                            new_result=model.fit(obs_vals,params=new_params,nan_policy='omit',method='powell',weights=weights)
                        else:
                            new_result=model.fit(obs_vals,params=new_params,nan_policy='omit',method='powell',weights=weights,**independent_vars)            
                        #Get the chisqr after fitting the new param
                        new_chisqr = calc_chisqr_val(obs_vals,new_result.best_fit,obs_errs)
                        param_list.append(param_val)
                        chisqr_list.append(new_chisqr)
                    #number of ponts below chisqrmax is the num_eval points
                    num_eval_points = np.sum([np.array(chisqr_list)<chisqrmax])
                    #num_eval_points == 0 or 1 is a special case. 
                    good_param_vals_i = [np.array(chisqr_list)<chisqrmax]
                    good_param_vals = np.array(param_list)[good_param_vals_i]
                    if num_eval_points>1:
                        max_point = np.max(good_param_vals)
                        min_point = np.min(good_param_vals)
                    else:
                        minus_points_i = [param_list<opt_val]
                        minus_points = np.array(param_list)[minus_points_i]
                        min_point = np.max(minus_points)
                        plus_points = np.array(param_list)[param_list>opt_val]
                        max_point = np.max(plus_points)
                        if debug==True:
                            print('new min point')
                            print(min_point)
                            print('new max point')
                            print(max_point)
                    if num_eval_points<num_test_points:
                        print('Insufficient number of points under max chisqr. Recursively iterating.')
                        print('Good points: '+str(num_eval_points)+'/'+str(num_test_points))
                #In theory should have all points needed to get uncertainties now.
                chisqr_list = np.array(chisqr_list)
                param_list = np.array(param_list) 

                #If there are new points that have a lower chisqr than the initial fit, 
                #   adjust the maxchisqr now or the error will be artificially large.
                min_eval_chisqr =  param_list[np.argmin(chisqr_list)]
                if min_eval_chisqr<chisqr0-np.abs(chisqr0-chisqrmax)/2.0:
                    print('WARNING: Local minima found that is different from initial value.')
                    print('Enable show_plots=True and check quality of initial fit.')
                    opt_val = param_list[np.argmin(chisqr_list)]
                    opt_chisqr  = np.mean([chisqr0,np.min(chisqr_list)])
                else:
                    opt_chisqr = chisqr0 
                    opt_val = init_param_val 
                temp_chisqrmin=opt_chisqr/(1.0+1.0/(num_points-num_free_params))
                temp_chisqrmax=opt_chisqr/(1.0-1.0/(num_points-num_free_params))  
                if extrapolate==True:
                    def parabola(x,a,b,c):
                        return a*((x-b)**2)+c
                    para_model = Model(parabola)
                    para_params = para_model.make_params()
                    #very roughly assume it's linear between 0 and 1 points
                    if prev_slope==False:
                        guess_slope=np.abs(temp_chisqrmax-chisqr0)/(((np.nanmax(np.array(param_list))-opt_val)**2))
                    else:
                        guess_slope = prev_slope
                    para_params.add('a',value=guess_slope,min=0,max=1e8)
                    para_params.add('b',vary=True,value=opt_val) #Should just be hte optimum value
                    para_params.add('c',vary=True,value=chisqr0,min=chisqr0-0.1,max=chisqrmax)
                    para_weights = np.ones(len(chisqr_list))
                    #weight by distance from optimum value?
                    para_weights = np.exp(-1.0*(np.abs(param_list-opt_val))/np.abs(opt_val))
                    para_fit = para_model.fit(chisqr_list,x=param_list,params=para_params,method='powell',weights=para_weights)
                    a_fit = para_fit.params['a'].value
                    b_fit = para_fit.params['b'].value
                    c_fit = para_fit.params['c'].value
                    prev_slope = a_fit
                    max_param_val = np.sqrt(np.abs((temp_chisqrmax-c_fit)/a_fit))+b_fit
                    #print('Max param_val='+str(max_param_val))
                    error = np.abs(max_param_val-b_fit)
                    opt_val=b_fit
                    eval_range = np.linspace(opt_val-error*1.2,opt_val+error*1.2,3000)
                    fit_eval = para_model.eval(x=eval_range,params=para_fit.params)
                    if affect_chisqr==True:
                        err_out[param]=error
                    else:
                        err_out[param]=np.nanmean(param_list)/np.sqrt(len(param_list))
                else:
                    good_param_val_i = [np.array(chisqr_list)<temp_chisqrmax]
                    good_param_vals = param_list[good_param_val_i]
                    good_chisqrs = chisqr_list[good_param_val_i]
                    max_i = np.argmax(good_chisqrs)
                    init_i =np.argmin(np.abs(np.max(good_chisqrs)-chisqr0))
                    error = np.abs(good_param_vals[max_i]-good_param_vals[init_i])
                    if affect_chisqr==True:
                        err_out[param]=error
                    else:
                        err_out[param]=np.nanmean(param_list)/np.sqrt(len(param_list))            
                if show_plots==True:
                    try:
                        plt.figure()
                        plt.plot(param_list,chisqr_list,color='k',marker='o',ls='')
                        plt.xlabel('Param val')
                        plt.ylabel('Chisqr')
                        plt.title('Uncertainty '+str(param)+ ' Error ='+str(round(error,3))+' or '+str(round(100*error/opt_val,3))+'%')
                        #plt.xlim(np.min(test_value_arr)-np.abs(np.min(test_value_arr))/10.0,np.max(test_value_arr)*1.1)
                        plt.plot(np.linspace(np.min(param_list),np.max(param_list)+1e-9,10),np.ones(10)*np.abs(temp_chisqrmax),'r')
                        if extrapolate==True:
                            plt.plot(eval_range,fit_eval,'b--')
                        plt.plot(opt_val+error,temp_chisqrmax,'g^')
                        plt.plot(opt_val-error,temp_chisqrmax,'g^')
                        plt.ylim(np.min(chisqr_list)-np.abs(np.abs(temp_chisqrmax)-np.min(chisqr_list))/3.0,np.abs(temp_chisqrmax)+np.abs(np.abs(temp_chisqrmax)-np.min(chisqr_list))/3.0)
                        #plt.xlim(0.9*np.min(test_value_arr),np.max(test_value_arr)*1.1)
                        plt.show()
                    except Exception as e:
                        print('Some error while plotting.')
                        print(e)
                    #plt.ylim(0.0,1.3*np.abs(chisqrmax))
                progress.current+=1
                progress()
            else:
                err_out[param]=0.0
        except Exception as e:
            err_out[param]=0
            print('Warning: Error when evaluating uncertainty. ')
            print(e)
    #Save to a file
    np.save(fname,err_out)
    return err_out