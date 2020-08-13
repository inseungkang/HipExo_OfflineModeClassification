import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import math
import os

fe_dir = "C:\\Users\\ikang7\\Dropbox (GaTech)\\ML\\data\\sensor_fusion\\feature extraction data\\"
# fe_dir = "C:/Users/ikang7/Dropbox (GaTech)/ML/data/sensor_fusion/feature extraction data/"
window_size = 350
transition_point = 0.2

trial_pool = [1]
subject_pool = [6]
mode_pool = ["RA2"]
# mode_pool = ["RA2", "RA3", "RA4", "RA5", "RD2", "RD3", "RD4", "RD5","SA1", "SA2", "SA3", "SA4", "SD1", "SD2", "SD3", "SD4"]
leg_pool = ["R"]
# subject_pool = [6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 27 ,28]

X_train = pd.DataFrame()
Y_train = pd.DataFrame()
gp_train = pd.DataFrame()

for trial in trial_pool:
    for subject in subject_pool:
        for mode in mode_pool:
            for starting_leg in leg_pool:
                data_path = fe_dir+"AB"+str(subject)+"_"+str(mode)+"_W"+str(window_size)+"_TP"+str(int(transition_point*10))+"_S2_"+str(starting_leg)+str(trial)+".csv"

                if path.exists(data_path) == 1:
                    for data_read_path in glob.glob(data_path):
                        data = pd.read_csv(data_read_path, header=None)
                        X = data.iloc[:, :-3]
                        Y = data.iloc[:, -1]
                        gp = data.iloc[:,-2]
                        X_train = pd.concat([X_train, X], axis=0, ignore_index=True)
                        Y_train = pd.concat([Y_train, Y], axis=0, ignore_index=True)
                        gp_train = pd.concat([gp_train, gp], axis=0, ignore_index=True)

        # train_path = fe_dir+"AB"+str(subject)+"_LG_W"+str(window_size)+"_TP0_S2_R"+str(trial)+".csv"    
        # if path.exists(train_path) == 1:
        #     for train_read_path in glob.glob(train_path):
        #         data = pd.read_csv(train_read_path, header=None)
        #         X = data.iloc[:, :-3]
        #         Y = data.iloc[:, -1]
        #         gp = data.iloc[:,-2]
        #         X_train = pd.concat([X_train, X], axis=0, ignore_index=True)
        #         Y_train = pd.concat([Y_train, Y], axis=0, ignore_index=True)
        #         gp_train = pd.concat([gp_train, gp], axis=0, ignore_index=True)