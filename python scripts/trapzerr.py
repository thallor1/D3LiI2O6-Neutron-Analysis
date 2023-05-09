import numpy as np

def get_trapz_err(x,y,errs,xlim=False):
    #Given the x values, errors, and limits of a trapzoidal integral returns the error bar of the 
    # result that would be given by np.trapz
    if xlim==False:
        xlim=[np.nanmin(x)-0.1,np.nanmax(x)+0.1]
    integral=0
    int_err=0
    good_i = np.intersect1d(np.where(x>=xlim[0])[0],np.where(x<=xlim[1])[0])
    x=x[good_i]
    errs=errs[good_i]
    int_err=0
    for i in range(len(errs)-1):
        yterm = 0.5*(y[i+1]+y[i])
        xterm = x[i+1]-x[i]
        yerr = np.sqrt(0.5*(errs[i]**2+errs[i+1]**2))
        z=xterm*yterm 
        zerr = np.sqrt(z**2*(yerr/yterm))
        int_err = np.sqrt(int_err**2 + zerr**2)
    return int_err