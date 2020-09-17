import pywt
from statsmodels.robust import mad
import numpy as np
import pandas as pd

def waveletSmooth( x, wavelet="haar", level=1, DecLvl=2, title=None):
    '''Using wavedec method, we can take coeff variables.
    These coeff variables contain a low coeff and the DecLvl number of high coeffs.
    i.e. [coeff1(122), coeff2(122), coeff3(244)]
    level variable means a index in which sigma will be calculated. And time series are perfectly denoised at selected level
    i.e. [coeff1(1245,2163,...), coeff2(243,0,123,...), coeff3(0,0,0,...)
    coeff[1:] means only high coeffs'''
    if type(x)==pd.DataFrame:
        denoised_columns_list = []
        for column in x.columns:
            value = x.loc[:,column]
            # calculate the wavelet coefficients
            coeff = pywt.wavedec( value, wavelet, mode="per", level=DecLvl )
            # calculate a threshold
            sigma = mad( coeff[-level] )
            # changing this threshold also changes the behavior,
            # but I have not played with this very much
            uthresh = sigma * np.sqrt( 2*np.log( len( value ) ) )
            coeff[1:] = (pywt.threshold(i, value=uthresh, mode="soft" ) for i in coeff[1:] )
            # reconstruct the signal using the thresholded coefficients
            y = pywt.waverec( coeff, wavelet, mode="per" )
            denoised_columns_list.append(list(y))

        denoised_columns_array = np.array(denoised_columns_list)
        denoised_columns_array = denoised_columns_array.transpose()
        return denoised_columns_array

    else:
        # calculate the wavelet coefficients
        coeff = pywt.wavedec(x, wavelet, mode="per", level=DecLvl)
        # calculate a threshold
        sigma = mad(coeff[-level])
        # changing this threshold also changes the behavior,
        # but I have not played with this very much
        uthresh = sigma * np.sqrt(2 * np.log(len(x)))
        coeff[1:] = (pywt.threshold(i, value=uthresh, mode="soft") for i in coeff[1:])
        # reconstruct the signal using the thresholded coefficients
        y = pywt.waverec(coeff, wavelet, mode="per")

        return y
