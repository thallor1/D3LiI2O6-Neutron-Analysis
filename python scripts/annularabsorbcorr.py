import numpy as np
from mantid.simpleapi import *


def annularabsorbcorr(f_list, Q_slice, E_slice, mat_string, Ei, outer_r, inner_r, samp_thick, samp_h,
                       num_density, eventnorm=True, wsname="Temp"):
    """
    Function for the correction of annular absorption. This requires reloading the file itself.

    :param mat_string: Material string in the format specified in Mantid documentation.
    :param outer_r: Outer annulus radius in cm
    :param inner_r: Inner annulus radius in cm
    :param samp_thick: thickness of sample in cm
    :param samp_h: height of sample in cm
    :param num_density: Number density of sample in f.u. / Ang^3
    :param eventnorm: Specifies if intensities should be normalized to monitor at the end.
    :return:
    """
    # First backup the pre-absorption corrected mdhistoworkspace.
    #CloneWorkspace(self.mdhisto, OutputWorkspace=self.name + '_noabs')
    output_ws_name = wsname + "_absorbcorr"
    # First load the data
    Load(Filename=f_list, OutputWorkspace=output_ws_name)
    load_ws = mtd[output_ws_name]
    merged_ws = MergeRuns(load_ws)
    # Convert to wavelength
    wavelength_ws_name = output_ws_name + '_wavelength'
    ConvertUnits(InputWorkspace=merged_ws, OutputWorkspace=wavelength_ws_name,
                 Target='Wavelength', EMode='Direct', EFixed=Ei)
    # Run Absorption Utility
    abs_ws_name = output_ws_name + '_ann_abs'
    wavelengthws = mtd[wavelength_ws_name]
    factors = AnnularRingAbsorption(InputWorkspace=wavelengthws, OutputWorkspace=abs_ws_name,
                                    CanOuterRadius=outer_r, CanInnerRadius=inner_r,
                                    SampleHeight=samp_h, SampleThickness=samp_thick,
                                    SampleChemicalFormula=mat_string, SampleNumberDensity=num_density)
    wavelengthws_corr = wavelengthws / factors

    # Convert back to Q
    abs_meV_ws = output_ws_name + '_ann_abs_meV'
    ConvertUnits(wavelengthws_corr, OutputWorkspace=abs_meV_ws, Target='DeltaE', Efixed=Ei,
                 Emode='Direct')
    working_ws = mtd[abs_meV_ws]

    # Convert to MD
    ws_corrected = ConvertToMD(working_ws, Qdimensions='|Q|')
    # Bin according to specified Q, E spacing
    outMD = BinMD(ws_corrected, AxisAligned=True, AlignedDim0=Q_slice, AlignedDim1=E_slice)
    # Normalize by num events
    nevents = outMD.getNumEventsArray()
    if eventnorm is False:
        pass
    else:
        I = np.copy(outMD.getSignalArray())
        Err = np.sqrt(np.copy(outMD.getErrorSquaredArray()))
        I /= nevents
        Err /= nevents
        outMD.setSignalArray(I)
        outMD.setErrorSquaredArray(Err ** 2)
    outMD = outMD.clone()
    return outMD

def annularabsorbcorrMD(f_list, mat_string, Ei, outer_r, inner_r, samp_thick, samp_h,
                       num_density, wsname="Temp"):
    """
    Function for the correction of annular absorption. This requires reloading the file itself.

    :param mat_string: Material string in the format specified in Mantid documentation.
    :param outer_r: Outer annulus radius in cm
    :param inner_r: Inner annulus radius in cm
    :param samp_thick: thickness of sample in cm
    :param samp_h: height of sample in cm
    :param num_density: Number density of sample in f.u. / Ang^3
    :param eventnorm: Specifies if intensities should be normalized to monitor at the end.
    :return:
    """
    # First backup the pre-absorption corrected mdhistoworkspace.
    #CloneWorkspace(self.mdhisto, OutputWorkspace=self.name + '_noabs')
    output_ws_name = wsname + "_absorbcorr"
    # First load the data
    Load(Filename=f_list, OutputWorkspace=output_ws_name)
    load_ws = mtd[output_ws_name]
    merged_ws = MergeRuns(load_ws)
    # Convert to wavelength
    wavelength_ws_name = output_ws_name + '_wavelength'
    ConvertUnits(InputWorkspace=merged_ws, OutputWorkspace=wavelength_ws_name,
                 Target='Wavelength', EMode='Direct', EFixed=Ei)
    # Run Absorption Utility
    abs_ws_name = output_ws_name + '_ann_abs'
    wavelengthws = mtd[wavelength_ws_name]
    factors = AnnularRingAbsorption(InputWorkspace=wavelengthws, OutputWorkspace=abs_ws_name,
                                    CanOuterRadius=outer_r, CanInnerRadius=inner_r,
                                    SampleHeight=samp_h, SampleThickness=samp_thick,
                                    SampleChemicalFormula=mat_string, SampleNumberDensity=num_density)
    wavelengthws_corr = wavelengthws / factors

    # Convert back to Q
    abs_meV_ws = output_ws_name + '_ann_abs_meV'
    ConvertUnits(wavelengthws_corr, OutputWorkspace=abs_meV_ws, Target='DeltaE', Efixed=Ei,
                 Emode='Direct')
    working_ws = mtd[abs_meV_ws]

    # Convert to MD
    ws_corrected = ConvertToMD(working_ws, Qdimensions='|Q|')
    # Bin according to specified Q, E spacing
    # outMD = BinMD(ws_corrected, AxisAligned=True, AlignedDim0=Q_slice, AlignedDim1=E_slice)
    outMD = ws_corrected.clone()
    outMD = outMD.clone()
    return outMD