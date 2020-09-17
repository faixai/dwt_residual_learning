# Imports (External)
import numpy as np
import pandas as pd
import copy
import sys
sys.path.append('../..')  
from sklearn import preprocessing
#Internal Imports
#from utils import pickle_load,pickle_save


def scale_periods(dict_dataframes):

    ddi_scaled = dict()
    scaler_parameters_subdict = {}

    for key, index_name in enumerate(dict_dataframes):
        ddi_scaled[index_name] = copy.deepcopy(dict_dataframes[index_name])

    for key, index_name in enumerate(ddi_scaled): 

        scaler = preprocessing.MinMaxScaler()

        for index,value in enumerate(ddi_scaled[index_name]):
            X_train = ddi_scaled[index_name][value][1]
            X_train_close = X_train.close.copy()
            fitted = scaler.fit(X_train)
            X_train_scaled = scaler.transform(X_train)
            X_train_scaled_df = pd.DataFrame(X_train_scaled,columns=list(X_train.columns))

            
            X_val = ddi_scaled[index_name][value][2]
            X_val_close = X_val.close.copy()
            X_val_scaled = (X_val - fitted.data_min_) / fitted.data_range_
            #X_val_scaled = scaler.transform(X_val)
            X_val_scaled_df = pd.DataFrame(X_val_scaled,columns=list(X_val.columns))

            
            X_test = ddi_scaled[index_name][value][3]
            X_test_close = X_test.close.copy()
            X_test_scaled = (X_test - fitted.data_min_) / fitted.data_range_
            #X_test_scaled = scaler.transform(X_test)
            X_test_scaled_df = pd.DataFrame(X_test_scaled,columns=list(X_test.columns))
            
            ddi_scaled[index_name][value][1] = X_train_scaled_df
            ddi_scaled[index_name][value][2] = X_val_scaled_df
            ddi_scaled[index_name][value][3] = X_test_scaled_df

            ddi_scaled[index_name][value]['raw_train_close'] = X_train_close
            ddi_scaled[index_name][value]['raw_val_close'] = X_val_close
            ddi_scaled[index_name][value]['raw_test_close'] = X_test_close

            # train data min, max, range save to inverse_transform later
            ddi_scaled[index_name][value]['min'] = fitted.data_min_
            ddi_scaled[index_name][value]['max'] = fitted.data_max_
            ddi_scaled[index_name][value]['range'] = fitted.data_range_

    return ddi_scaled

if __name__ == '__main__':
    dict_dataframes_index=pickle_load(path_filename="../data/interim/cdii_tvt_split.pickle")
    print("scale_dataset - Start...")
    ddi_scaled = scale_periods(dict_dataframes_index)
    pickle_save(ddi_scaled,path_filename="../data/interim/cdii_tvt_split_scaled")
    print("scale_dataset - Finished.")