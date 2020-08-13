import pandas as pd 
import sys
import os
import glob
import numpy as  np 
import matplotlib.pyplot as plt
import math
from joblib import Parallel, delayed
from sklearn.preprocessing import MinMaxScaler
from scipy import signal

def combine_feature(subject, trial_mode, window_size, starting_leg, transition_point, trial_number, data_type):
    base_path = "/HDD/hipexo/Dropbox (GaTech)/ML/data/mode_classification/feature_data/"

    MECH_R_data = pd.DataFrame()
    MECH_L_data = pd.DataFrame()
    EMG_R_data = pd.DataFrame()
    EMG_L_data = pd.DataFrame()
    combine_feature_data = pd.DataFrame()

    if trial_number in [1, 2, 3, 4]:
        MECH_R_data_path = base_path+"AB0"+str(subject)+"/AB0"+str(subject)+"_"+str(trial_mode)+"_W"+str(window_size*10)+"_TP"+str(transition_point)+"_"+str(starting_leg)+str(trial_number)+"_MECH.csv"
        EMG_R_data_path = base_path+"AB0"+str(subject)+"/AB0"+str(subject)+"_"+str(trial_mode)+"_W"+str(window_size*10)+"_TP"+str(transition_point)+"_"+str(starting_leg)+str(trial_number)+"_EMG.csv"

        for MECH_R_read_path in glob.glob(MECH_R_data_path):
            MECH_R_data = pd.read_csv(MECH_R_read_path, header=None)
        for EMG_R_data_path in glob.glob(EMG_R_data_path):
            EMG_R_data = pd.read_csv(EMG_R_data_path, header=None)
        combine_feature_data = pd.concat([EMG_R_data.iloc[:,:32], MECH_R_data], axis=1, ignore_index=True) 

    if trial_number in [5, 6, 7, 8]:
        MECH_L_data_path = base_path+"AB0"+str(subject)+"/AB0"+str(subject)+"_"+str(trial_mode)+"_W"+str(window_size*10)+"_TP"+str(transition_point)+"_"+str(starting_leg)+str(trial_number)+"_MECH.csv"
        EMG_L_data_path = base_path+"AB0"+str(subject)+"/AB0"+str(subject)+"_"+str(trial_mode)+"_W"+str(window_size*10)+"_TP"+str(transition_point)+"_"+str(starting_leg)+str(trial_number)+"_EMG.csv"

        for MECH_L_data_path in glob.glob(MECH_L_data_path):
            MECH_L_data = pd.read_csv(MECH_L_data_path, header=None)
        for EMG_L_data_path in glob.glob(EMG_L_data_path):
            EMG_L_data = pd.read_csv(EMG_L_data_path, header=None)
        combine_feature_data = pd.concat([EMG_L_data.iloc[:,-32:], MECH_L_data], axis=1, ignore_index=True) 

    save_path = base_path+"AB0"+str(subject)+"/AB0"+str(subject)+"_"+str(trial_mode)+"_W"+str(window_size*10)+"_TP"+str(transition_point)+"_"+str(starting_leg)+str(trial_number)+"_EMG_"+str(data_type)+".csv"
    print("AB0"+str(subject)+"_"+str(trial_mode)+"_W"+str(window_size*10)+"_"+str(starting_leg)+"_"+str(trial_number))
    combine_feature_data.to_csv(save_path, sep=',', index=False, header=False)

for subject in [1, 2]:
    for trial_mode in ["L0", "RA1", "RA2", "RA3", "RA4", "RD1", "RD2", "RD3", "RD4", "SA1", "SA2", "SA3", "SA4", "SD1", "SD2", "SD4"]:
        for window_size in [15]:
            for starting_leg in ["R", "L"]:
                for trial_number in [1, 2, 3, 4, 5, 6, 7, 8]:
                    for transition_point in [0, 2000, 4000]:
                        for data_type in ["COMBINED"]:
                            combine_feature(subject, trial_mode, window_size, starting_leg, transition_point, trial_number, data_type)
