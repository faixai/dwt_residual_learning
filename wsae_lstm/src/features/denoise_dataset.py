# Imports (External)
import numpy as np
import pandas as pd
import copy
import sys
sys.path.append('../..')  
#Internal Imports
from src.utils import pickle_load,pickle_save
from subrepos.DeepLearning_Financial.models.wavelet import waveletSmooth

def denoise_periods(dict_dataframes):

    ddi_denoised= dict() 

    for key, index_name in enumerate(dict_dataframes):
        ddi_denoised[index_name] = copy.deepcopy(dict_dataframes[index_name])

    for key, index_name in enumerate(ddi_denoised): 
        for index,value in enumerate(ddi_denoised[index_name]):
            
            X_train_scaled = ddi_denoised[index_name][value][1]
            X_val_scaled = ddi_denoised[index_name][value][2]
            X_test_scaled = ddi_denoised[index_name][value][3]

            X_train_scaled_denoised_df = pd.DataFrame(waveletSmooth(X_train_scaled),columns=list(X_train_scaled.columns))      
            X_val_scaled_denoised_df = pd.DataFrame(waveletSmooth(X_val_scaled),columns=list(X_val_scaled.columns))  
            X_test_scaled_denoised_df = pd.DataFrame(waveletSmooth(X_test_scaled),columns=list(X_test_scaled.columns))      

            if len(X_train_scaled) != len(X_train_scaled_denoised_df):
                X_train_scaled_denoised_df = X_train_scaled_denoised_df.drop(len(X_train_scaled_denoised_df)-1,0)

            if len(X_val_scaled) != len(X_val_scaled_denoised_df):
                X_val_scaled_denoised_df = X_val_scaled_denoised_df.drop(len(X_val_scaled_denoised_df)-1,0)

            if len(X_test_scaled) != len(X_test_scaled_denoised_df):
                X_test_scaled_denoised_df = X_test_scaled_denoised_df.drop(len(X_test_scaled_denoised_df)-1,0)

            train_nan_index = pd.isna(X_train_scaled_denoised_df)
            val_nan_index = pd.isna(X_val_scaled_denoised_df)
            test_nan_index = pd.isna(X_test_scaled_denoised_df)

            X_train_scaled_denoised_df = pd.DataFrame(
                np.where(train_nan_index, X_train_scaled, X_train_scaled_denoised_df),
                columns=list(X_train_scaled_denoised_df))

            X_val_scaled_denoised_df = pd.DataFrame(
                np.where(val_nan_index, X_val_scaled, X_val_scaled_denoised_df),
                columns=list(X_val_scaled_denoised_df))

            X_test_scaled_denoised_df = pd.DataFrame(
                np.where(test_nan_index, X_test_scaled, X_test_scaled_denoised_df),
                columns=list(X_test_scaled_denoised_df))



            ddi_denoised[index_name][value][1] = X_train_scaled_denoised_df
            ddi_denoised[index_name][value][2] = X_val_scaled_denoised_df 
            ddi_denoised[index_name][value][3] = X_test_scaled_denoised_df    

    return ddi_denoised


if __name__ == '__main__':
    print("denoise_dataset - Start...")
    ddi_scaled=pickle_load(path_filename="../data/interim/cdii_tvt_split_scaled.pickle")
    ddi_denoised= denoise_periods(ddi_scaled)
    pickle_save(ddi_denoised,path_filename="../data/interim/cdii_tvt_split_scaled_denoised")
    print("denoise_dataset - Finished.")