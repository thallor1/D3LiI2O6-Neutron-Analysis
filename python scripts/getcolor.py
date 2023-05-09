import matplotlib

def get_color_from_value(val,vmin,vmax,cmap):
    #Function to change a colormap smoothly according to a value. 
    #Min/max values are important to define
    cmap=matplotlib.cm.get_cmap(cmap)
    norm_val = matplotlib.colors.Normalize(vmin=vmin,vmax=vmax)
    rgba = cmap(norm_val(val))
    return rgba