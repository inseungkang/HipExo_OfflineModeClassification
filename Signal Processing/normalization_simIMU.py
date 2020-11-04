import pandas as pd 
import sys
import os
import glob
import numpy as  np 
import matplotlib.pyplot as plt
import math
from os import path
from joblib import Parallel, delayed
from sklearn.preprocessing import MinMaxScaler
from scipy import signal

def normaliza_data(subject, folder):
    base_path_dir = "/HDD/hipexo/Inseung/sim IMU data/POSE" + str(folder) + "/"
    trial_pool = [1, 2, 3, 4, 5, 6]
    Data = pd.DataFrame()

    for trial in trial_pool:
        if subject > 9:
            data_path = base_path_dir+"AB"+str(subject)+"_LG_S2_T"+str(trial)+".csv"
        else:
            data_path = base_path_dir+"AB0"+str(subject)+"_LG_S2_T"+str(trial)+".csv"

        if path.exists(data_path) == 1:

            for data_read_path in glob.glob(data_path):
                data = pd.read_csv(data_read_path, header=0)
                X = data.iloc[:, :-3]
                Data = pd.concat([Data, X], axis=0, ignore_index=True)

    Data = Data.to_numpy(dtype = float)
    norm_mat = np.ones((len(Data[0]),2))

    # print(Data[:,0].mean())
    # print(Data[:,0].std())
    # # Data_new = (Data[:,0] - Data[:,0].min())/(Data[:,0].max()-Data[:,0].min())

    # plt.plot(Data_new)
    # plt.show()
    for idx in range(0, len(Data[0])):
        norm_mat[idx,0] = Data[:,idx].mean()
        norm_mat[idx,1] = Data[:,idx].std()
# 
    saving_data = pd.DataFrame(norm_mat)
    save_path = "/HDD/hipexo/Inseung/norm_matrix/"+"AB"+str(subject)+"_norm_IMU" + str(folder) +".csv"

    print(save_path)
    saving_data.to_csv(save_path, sep=',', index=False, header=False)

for folder in [1, 2, 3, 4, 5, 6, 7, 8, 9]:
    for subject in [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 27 ,28]:
        normaliza_data(subject, folder)



