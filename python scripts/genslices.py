def GenQslice(Qmin,Qmax,Qbins):
    QSlice = '|Q|, '+str(Qmin)+','+str(Qmax)+','+str(Qbins)
    return QSlice
def GenEslice(Emin,Emax,Ebins):
    Eslice = 'DeltaE, '+str(Emin)+','+str(Emax)+','+str(Ebins)
    return Eslice