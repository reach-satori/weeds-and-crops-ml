import numpy as np
from matplotlib import pyplot as plt
import cv2
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


def ndvi_index(img):
    '''
    normalized difference vegetation index
    '''
    ch_nir, ch_red = img[...,4], img[...,0]
    return (ch_nir-ch_red)/(ch_nir+ch_red)


def ndvi_index_with_threshhold(img, thresh=0.4):
    ndvi = ndvi_index(img)

    return np.where(ndvi > thresh, 1., ndvi)


def dvi_index(img):
    '''
    sensitive to soil background
    '''
    ch_nir, ch_red = img[...,4], img[...,0]
    return ch_nir-ch_red


def rvi_index(img):
    '''
    sensitive to soil background
    '''
    ch_nir, ch_red = img[...,4], img[...,0]
    return ch_nir/ch_red


def msr_index(img):
    '''
    improved vegetation sensitivity
    '''
    ch_nir, ch_red = img[...,4], img[...,0]
    ratio = ch_nir/ch_red
    return (ratio - 1)/np.sqrt(ratio + 1)


def tvi_index(img):
    '''
    modifies NDVI with only positive values
    < 0.71 as non-vegetation
    > 0.71 as vegetation
    '''
    ch_nir, ch_red = img[...,4], img[...,0]
    ratio = ch_nir/ch_red
    return np.sqrt((ratio - 1)/(ratio + 1)+0.5)


# def mtvi_index(img, c):
#     '''
#     used with poor vegtation
#     '''
#     ch_nir, ch_red = img[...,4], img[...,0]
#     return np.sqrt((c * ch_nir - ch_red)/(c * ch_nir + ch_red))


def gndvi_index(img):
    '''
    green NDVI
    '''
    ch_nir, ch_green = img[...,4], img[...,1]
    return (ch_nir-ch_green)/(ch_nir+ch_green)


def gci_index(img):
    '''
    Green Chlorophyll Index
    '''
    ch_nir, ch_green = img[...,4], img[...,1]
    return (ch_nir/ch_green) - 1


def savi_index(img):
    '''
    Soil Adjusted Vegetation Index
    '''
    ch_nir, ch_red = img[...,4], img[...,0]
    return 1.5 * (ch_nir - ch_red) / (ch_nir + ch_red + 0.5)


def evi_index(img):
    '''
    Enhanced Vegetation Index
    '''
    ch_nir, ch_red, ch_blue = img[...,4], img[...,0], img[...,2]
    return 2.5 * (ch_nir - ch_red) / (1 + ch_nir + 6 * ch_red - 7.5 * ch_blue)


def sr_index(img):
    '''
    simple ratio
    '''
    ch_nir, ch_red = img[...,4], img[...,0]
    return ch_nir/ch_red


def cl_g_index(img):
    '''
    green chlorophyll index
    '''
    ch_nir, ch_green = img[...,4], img[...,1]
    return ch_nir - ch_green - 1

def index_viewer(img):
    indexlist = ["ndvi_index",
                 "ndvi_index_with_threshhold",
                 "dvi_index",
                 "rvi_index",
                 "msr_index",
                 "tvi_index",
                 "gndvi_index",
                 "gci_index",
                 "savi_index",
                 "evi_index",
                 "sr_index",
                 "cl_g_index"]

    img = cv2.normalize(img, img, 0, 1, cv2.NORM_MINMAX)
    for i, ind in enumerate(indexlist):
        plt.subplot(4, 3, i+1)  # assuming 12 indices
        plt.title(ind)
        plt.xticks([])
        plt.yticks([])
        imc = img.copy()
        indimg = eval(f"{ind}(imc)")
        indimg = cv2.normalize(indimg, indimg, 0, 1, cv2.NORM_MINMAX)
        plt.imshow(indimg.T, cmap="viridis")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    from load_images import load_images
    for img in load_images():
        index_viewer(img)
