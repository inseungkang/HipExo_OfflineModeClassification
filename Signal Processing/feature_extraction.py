import pandas as pd 
import sys
import os
from os import path
import glob
import numpy as np 
import matplotlib.pyplot as plt
import math
from joblib import Parallel, delayed

def feature_extraction(data, window_size):
    frame_rate = 5
    window_size = int(window_size/frame_rate)
    feature_extracted_data = pd.DataFrame()
    feature_extraction_columns = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    slinding_rate = 10
    for i, column in enumerate(data.columns):
        if i in feature_extraction_columns:

            single_column = data.iloc[:,i].values
            shape_des = single_column.shape[:-1] + (single_column.shape[-1] - window_size + 1, window_size)
            strides_des = single_column.strides + (single_column.strides[-1],)
            
            sliding_window = np.lib.stride_tricks.as_strided(single_column, shape=shape_des, strides=strides_des)[::slinding_rate]
            sliding_window_df = pd.DataFrame(sliding_window)

            min_series = sliding_window_df.min(axis=1)
            max_series = sliding_window_df.max(axis=1)
            mean_series = sliding_window_df.mean(axis=1)
            std_series = sliding_window_df.std(axis=1)
            start_series = sliding_window_df.iloc[:,0]
            
            feature_extracted_data = pd.concat([feature_extracted_data, round(min_series,4), round(max_series,4), round(mean_series,4), round(std_series,4), round(start_series,4)], axis=1, ignore_index=True)

    mode_column = data.iloc[window_size-1: , -1].reset_index(drop=True)[::slinding_rate].reset_index(drop=True)
    gait_phase = data.iloc[window_size-1: , -2].reset_index(drop=True)[::slinding_rate].reset_index(drop=True)
    true_mode = data.iloc[window_size-1: , -3].reset_index(drop=True)[::slinding_rate].reset_index(drop=True)

    feature_extracted_data = pd.concat([feature_extracted_data, true_mode, round(gait_phase,2), mode_column], axis=1, ignore_index=True)
    return feature_extracted_data
def fe_parallel(combo):
    subject = combo[0]
    window_size = combo[1]
    trial_mode = combo[2]
    starting_leg = combo[3]
    transition_point = combo[4]
    walking_speed = combo[5]
    trial_number = combo[6]

    # base_path = "/HDD/Inseung/Dropbox (GaTech)/ML/data/sensor_fusion/Opensim Data/"
    # norm_path = "/HDD/Inseung/Dropbox (GaTech)/ML/data/sensor_fusion/norm_matrix/AB"+str(subject)+"_norm.csv"
    base_path = "C:\\Users\\ikang7\\Dropbox (GaTech)\\ML\\data\\sensor_fusion\\Opensim Data\\"
    norm_path = "C:\\Users\\ikang7\\Dropbox (GaTech)\\ML\\data\\sensor_fusion\\norm_matrix\\AB"+str(subject)+"_norm.csv"

    if subject > 9:
        if trial_mode == "LG":
            data_path = base_path+"AB"+str(subject)+"_"+str(trial_mode)+"_S"+str(walking_speed)+"_T"+str(trial_number)+".csv"
        else:
            data_path = base_path+"AB"+str(subject)+"_"+str(trial_mode)+"_S2_"+str(starting_leg)+"_T"+str(trial_number)+".csv"
    else:
        if trial_mode == "LG":
            data_path = base_path+"AB0"+str(subject)+"_"+str(trial_mode)+"_S"+str(walking_speed)+"_T"+str(trial_number)+".csv"
        else:
            data_path = base_path+"AB0"+str(subject)+"_"+str(trial_mode)+"_S2_"+str(starting_leg)+"_T"+str(trial_number)+".csv"

    norm_mat = pd.read_csv(norm_path, header=None)
    norm_mat = norm_mat.to_numpy(dtype = float)

    if path.exists(data_path) == 1:
        for read_path in glob.glob(data_path):
            feature_extracted_data = pd.DataFrame()
            # data = pd.read_csv(read_path, header=0)
            # mode_column = data.iloc[:,-1].reset_index(drop=True)

            raw_data = pd.read_csv(read_path, header=0)
            processing_data = raw_data.iloc[:,:-3].reset_index(drop=True)
            processing_data = processing_data.to_numpy(dtype = float)          
            processing_data = (processing_data - norm_mat[:,0])/norm_mat[:,1]
            processing_data = pd.DataFrame(processing_data)
            data = pd.concat([processing_data, raw_data.iloc[:,-3:]], axis=1)            
            mode_column = raw_data.iloc[:,-1].reset_index(drop=True)

            if "LG" in read_path:
                feature_extracted_data = feature_extraction(data, window_size)
            
            else:
                mode_diff = np.diff(mode_column)
                transition_idx = []

                for i, mode_number in enumerate(mode_diff):
                    if abs(mode_number).any() != 0:
                        transition_idx.append(i+1)

                tran1_start_idx = 0
                tran1_end_idx = transition_idx[0]
                tran2_start_idx = transition_idx[1]
                tran2_end_idx = len(mode_column)-1

                diff1_idx = tran1_end_idx - tran1_start_idx + 1
                diff2_idx = tran2_end_idx - tran2_start_idx + 1

                partition1_idx = tran1_start_idx + int(math.ceil(diff1_idx * transition_point))
                partition2_idx = tran2_start_idx + int(math.ceil(diff2_idx * transition_point))
                
                walk_mode = 0
                change_mode = mode_column.iloc[tran1_end_idx + 1]

                mode_column.iloc[tran1_start_idx : partition1_idx] = walk_mode
                mode_column.iloc[partition1_idx : tran1_end_idx + 1] = change_mode
                mode_column.iloc[tran2_start_idx : partition2_idx] = change_mode
                mode_column.iloc[partition2_idx : tran2_end_idx + 1] = walk_mode

                data = pd.concat([data.iloc[:,:-1], mode_column], axis=1, ignore_index=True)
                feature_extracted_data = feature_extraction(data, window_size)

        # save_path_dir = "/HDD/Inseung/Dropbox (GaTech)/ML/data/sensor_fusion/feature extraction data/"
        save_path_dir = "C:\\Users\\ikang7\\Desktop\\transitional label sweep feature extraction\\"+str(window_size)+"\\"
        print("Extracting AB"+str(subject)+" "+str(trial_mode)+", Window Size "+str(window_size)+", Transition "+str(int(transition_point*100))+", Speed "+str(walking_speed)+", Trial Number "+str(starting_leg)+str(trial_number))

        if trial_mode == "LG":
            save_path = save_path_dir+"AB"+str(subject)+"_"+str(trial_mode)+"_W"+str(window_size)+"_TP"+str(int(transition_point*100))+"_S"+str(walking_speed)+"_R"+str(trial_number)+".csv"
            if starting_leg == "R":
                feature_extracted_data.to_csv(save_path, sep=',', index=False, header=False)
        else:
            save_path = save_path_dir+"AB"+str(subject)+"_"+str(trial_mode)+"_W"+str(window_size)+"_TP"+str(int(transition_point*100))+"_S2_"+str(starting_leg)+str(trial_number)+".csv"
            if walking_speed == 2:
                feature_extracted_data.to_csv(save_path, sep=',', index=False, header=False)

run_combos = []
for subject in [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 27 ,28, 30]:
    for window_size in [350, 750]:
        for trial_mode in ["LG"]:
            for starting_leg in ["R"]:
                for transition_point in [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]:
                    for walking_speed in [2]:
                        for trial_number in [1, 2, 3]:
                            run_combos.append([subject, window_size, trial_mode, starting_leg, transition_point, walking_speed, trial_number])
Parallel(n_jobs=-1)(delayed(fe_parallel)(combo) for combo in run_combos)

run_combos = []
for subject in [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 27 ,28, 30]:
    for window_size in [350, 750]:
        for trial_mode in ["RA2", "RA3", "RA4", "RA5", "RD2", "RD3", "RD4", "RD5", "SA1", "SA2", "SA3", "SA4", "SD1", "SD2", "SD3", "SD4"]:
            for starting_leg in ["R", "L"]:
                for transition_point in [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]:
                    for walking_speed in [2]:
                        for trial_number in [1, 2, 3]:
                            run_combos.append([subject, window_size, trial_mode, starting_leg, transition_point, walking_speed, trial_number])
Parallel(n_jobs=-1)(delayed(fe_parallel)(combo) for combo in run_combos)