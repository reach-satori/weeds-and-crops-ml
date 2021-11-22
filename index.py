import numpy as np
'''
Index List (13):

* NDVI 
* NDVI with thresh
* DVI 
* RVI 
* MSR
* TVI
* MTVI
* GNDVI
* GCI
* SAVI
* EVI
* SR
* CL-G


https://www.researchgate.net/publication/324161910_Use_of_Multispectral_Airborne_Images_to_Improve_In-Season_Nitrogen_Management_Predict_Grain_Yield_and_Estimate_Economic_Return_of_Maize_in_Irrigated_High_Yielding_Environments
https://www.indexdatabase.de/db/i-single.php?id=31
https://www.allacronyms.com/green_soil_adjusted_vegetation_index/abbreviated
https://www.sciencedirect.com/topics/earth-and-planetary-sciences/normalized-difference-vegetation-index
https://www.precisionhawk.com/precisionanalytics-agriculture
'''

def ndvi_index(ch_nir, ch_red): 
    '''
    normalized difference vegetation index
    '''
    return (ch_nir-ch_red)/(ch_nir+ch_red)


def ndvi_index_with_threshhold(ndvi_index, thresh):
    return np.where(ndvi_index > 0.4, 100, ndvi_index)


def dvi_index(ch_nir, ch_red): 
    '''
    sensitive to soil background
    '''
    return ch_nir-ch_red


def rvi_index(ch_nir, ch_red): 
    '''
    sensitive to soil background
    '''
    return ch_nir/ch_red


def msr_index(ch_nir, ch_red): 
    '''
    improved vegetation sensitivity
    '''
    ratio = ch_nir/ch_red 
    return (ratio - 1)/np.sqrt(ratio + 1)


def tvi_index(ch_nir, ch_red): 
    '''
    modifies NDVI with only positive values
    < 0.71 as non-vegetation 
    > 0.71 as vegetation
    '''
    ratio = ch_nir/ch_red 
    return np.sqrt((ratio - 1)/(ratio + 1)+0.5)


def mtvi_index(ch_nir, ch_red, c): 
    '''
    used with poor vegtation
    '''
    return np.sqrt((c * ch_nir - ch_red )/(c * ch_nir + ch_red ))


def gndvi_index(ch_nir, ch_green): 
    '''
    green NDVI
    '''
    return (ch_nir-ch_green)/(ch_nir+ch_green)


def gci_index(ch_nir, ch_green): 
    '''
    Green Chlorophyll Index
    '''
    return (ch_nir/ch_green) - 1


def savi_index(ch_nir, ch_red): 
    '''
    Soil Adjusted Vegetation Index
    '''
    return 1.5 * (ch_nir - ch_red) / (ch_nir + ch_red + 0.5)


def evi_index(ch_nir, ch_red, ch_blue): 
    '''
    Enhanced Vegetation Index 
    '''
    return 2.5 * (ch_nir - ch_red) / (1 + ch_nir + 6 * ch_red - 7.5 * ch_blue)


def sr_index(ch_nir, ch_red): 
    '''
    simple ratio
    '''
    return ch_nir/ch_red 


def cl_g_index(ch_nir, ch_green): 
    '''
    green chlorophyll index
    '''
    return ch_nir - ch_green - 1
