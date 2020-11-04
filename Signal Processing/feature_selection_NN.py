import pandas as pd
import numpy as np
import math
from sklearn import preprocessing
import glob
import os
from os import path
from sklearn.model_selection import train_test_split, cross_val_score, KFold
import collections
import statistics
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import xgboost as xgb
from joblib import Parallel, delayed
import gc
os.environ["CUDA_VISIBLE_DEVICES"] = '-1'

def xgboost_parallel(combo):
    channel = combo[0]
    testing_subject = combo[1]
    feature_list = chosen_features + [channel]

    trial_pool = [1, 2, 3]
    subject_pool = [6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 27 ,28]
    del subject_pool[subject_pool.index(testing_subject)]

    params = {'verbosity':0, 'objective':'multi:softmax', 'num_class':5, 'max_depth':tree_depth, 'min_child_weight':child_weight}
    X_train = pd.DataFrame()
    Y_train = pd.DataFrame()
    gp_train = pd.DataFrame()
    Y_test_result = []
    Y_pred_result = []

    for trial in trial_pool:
        for subject in subject_pool:
            for mode in ["RA2", "RA3", "RA4", "RA5", "RD2", "RD3", "RD4", "RD5","SA1", "SA2", "SA3", "SA4", "SD1", "SD2", "SD3", "SD4"]:
                for starting_leg in ["R", "L"]:
                    train_path = fe_dir+"AB"+str(subject)+"_"+str(mode)+"_W"+str(window_size)+"_TP"+str(int(transition_point*100))+"_S2_"+str(starting_leg)+str(trial)+".csv"

                    if path.exists(train_path) == 1:
                        for train_read_path in glob.glob(train_path):
                            data = pd.read_csv(train_read_path, header=None)
                            X = data.iloc[:, feature_list]
                            Y = data.iloc[:, -1]
                            gp = data.iloc[:,-2]
                            X_train = pd.concat([X_train, X], axis=0, ignore_index=True)
                            Y_train = pd.concat([Y_train, Y], axis=0, ignore_index=True)
                            gp_train = pd.concat([gp_train, gp], axis=0, ignore_index=True)

            train_path = fe_dir+"AB"+str(subject)+"_LG_W"+str(window_size)+"_TP0_S2_R"+str(trial)+".csv"    
            if path.exists(train_path) == 1:
                for train_read_path in glob.glob(train_path):
                    data = pd.read_csv(train_read_path, header=None)
                    X = data.iloc[:, feature_list]
                    Y = data.iloc[:, -1]
                    gp = data.iloc[:,-2]
                    X_train = pd.concat([X_train, X], axis=0, ignore_index=True)
                    Y_train = pd.concat([Y_train, Y], axis=0, ignore_index=True)
                    gp_train = pd.concat([gp_train, gp], axis=0, ignore_index=True)

    xg_train = xgb.DMatrix(X_train, label=Y_train)
    model = xgb.train(params, xg_train, num_boost_round = boost_round)
    del [[X, Y, gp, X_train, Y_train, gp_train]]

    for mode in ["RA2", "RA3", "RA4", "RA5", "RD2", "RD3", "RD4", "RD5", "SA1", "SA2", "SA3", "SA4", "SD1", "SD2", "SD3", "SD4"]:
        for starting_leg in ["R", "L"]:   
            for trial in trial_pool: 
                test_path = fe_dir+"AB"+str(testing_subject)+"_"+str(mode)+"_W"+str(window_size)+"_TP"+str(int(transition_point*100))+"_S2_"+str(starting_leg)+str(trial)+".csv"

                if path.exists(test_path) == 1:
                    for test_read_path in glob.glob(test_path):
                        data = pd.read_csv(test_read_path, header=None)
                        X = data.iloc[:, feature_list]
                        Y = data.iloc[:, -1]
                        xg_test = xgb.DMatrix(X, label=Y)
                        Y_pred = model.predict(xg_test)
                        Y_pred_result = np.concatenate((Y_pred_result, Y_pred))
                        Y_test_result = np.concatenate((Y_test_result, Y))

    for trial in trial_pool:
        train_path = fe_dir+"AB"+str(testing_subject)+"_LG_W"+str(window_size)+"_TP0_S2_R"+str(trial)+".csv"

        if path.exists(test_path) == 1:
            for test_read_path in glob.glob(test_path):
                data = pd.read_csv(test_read_path, header=None)
                X = data.iloc[:, feature_list]
                Y = data.iloc[:, -1]
                xg_test = xgb.DMatrix(X, label=Y)
                Y_pred = model.predict(xg_test)
                Y_pred_result = np.concatenate((Y_pred_result, Y_pred))
                Y_test_result = np.concatenate((Y_test_result, Y))
                del [[X, Y, Y_pred, xg_test]]

    Y_test_result = np.ravel(Y_test_result)
    Y_pred_result = np.ravel(Y_pred_result)
    xgboost_overall_accuracy = accuracy_score(Y_test_result, Y_pred_result)
    output = [channel, xgboost_overall_accuracy]
    return output
def lda_parallel(combo):
    channel = combo[0]
    testing_subject = combo[1]
    feature_list = chosen_features + [channel]

    trial_pool = [1, 2, 3]
    subject_pool = [6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 27 ,28]
    del subject_pool[subject_pool.index(testing_subject)]

    X_train = pd.DataFrame()
    Y_train = pd.DataFrame()
    gp_train = pd.DataFrame()
    Y_test_result = []
    Y_pred_result = []

    for trial in trial_pool:
        for subject in subject_pool:
            for mode in ["RA2", "RA3", "RA4", "RA5", "RD2", "RD3", "RD4", "RD5","SA1", "SA2", "SA3", "SA4", "SD1", "SD2", "SD3", "SD4"]:
                for starting_leg in ["R", "L"]:
                    train_path = fe_dir+"AB"+str(subject)+"_"+str(mode)+"_W"+str(window_size)+"_TP"+str(int(transition_point*100))+"_S2_"+str(starting_leg)+str(trial)+".csv"

                    if path.exists(train_path) == 1:
                        for train_read_path in glob.glob(train_path):
                            data = pd.read_csv(train_read_path, header=None)
                            X = data.iloc[:, feature_list]
                            Y = data.iloc[:, -1]
                            gp = data.iloc[:,-2]
                            X_train = pd.concat([X_train, X], axis=0, ignore_index=True)
                            Y_train = pd.concat([Y_train, Y], axis=0, ignore_index=True)
                            gp_train = pd.concat([gp_train, gp], axis=0, ignore_index=True)

            train_path = fe_dir+"AB"+str(subject)+"_LG_W"+str(window_size)+"_TP0_S2_R"+str(trial)+".csv"    
            if path.exists(train_path) == 1:
                for train_read_path in glob.glob(train_path):
                    data = pd.read_csv(train_read_path, header=None)
                    X = data.iloc[:, feature_list]
                    Y = data.iloc[:, -1]
                    gp = data.iloc[:,-2]
                    X_train = pd.concat([X_train, X], axis=0, ignore_index=True)
                    Y_train = pd.concat([Y_train, Y], axis=0, ignore_index=True)
                    gp_train = pd.concat([gp_train, gp], axis=0, ignore_index=True)

    lda_model = LDA()
    lda_model.fit(X_train, np.ravel(Y_train))
    del [[X, Y, gp, X_train, Y_train, gp_train]]

    for mode in ["RA2", "RA3", "RA4", "RA5", "RD2", "RD3", "RD4", "RD5", "SA1", "SA2", "SA3", "SA4", "SD1", "SD2", "SD3", "SD4"]:
        for starting_leg in ["R", "L"]:   
            for trial in trial_pool: 
                test_path = fe_dir+"AB"+str(testing_subject)+"_"+str(mode)+"_W"+str(window_size)+"_TP"+str(int(transition_point*100))+"_S2_"+str(starting_leg)+str(trial)+".csv"

                if path.exists(test_path) == 1:
                    for test_read_path in glob.glob(test_path):
                        data = pd.read_csv(test_read_path, header=None)
                        X = data.iloc[:, feature_list]
                        Y = data.iloc[:, -1]
                        Y_pred = lda_model.predict(X)
                        Y_pred_result = np.concatenate((Y_pred_result, Y_pred))
                        Y_test_result = np.concatenate((Y_test_result, Y))

    for trial in trial_pool:
        train_path = fe_dir+"AB"+str(testing_subject)+"_LG_W"+str(window_size)+"_TP0_S2_R"+str(trial)+".csv"

        if path.exists(test_path) == 1:
            for test_read_path in glob.glob(test_path):
                data = pd.read_csv(test_read_path, header=None)
                X = data.iloc[:, feature_list]
                Y = data.iloc[:, -1]
                Y_pred = lda_model.predict(X)
                Y_pred_result = np.concatenate((Y_pred_result, Y_pred))
                Y_test_result = np.concatenate((Y_test_result, Y))
                del [[X, Y, Y_pred]]

    Y_test_result = np.ravel(Y_test_result)
    Y_pred_result = np.ravel(Y_pred_result)
    LDA_overall_accuracy = accuracy_score(Y_test_result, Y_pred_result)
    output = [channel, LDA_overall_accuracy]
    return output
def SVM_parallel(combo):
    channel = combo[0]
    testing_subject = combo[1]
    feature_list = chosen_features + [channel]

    trial_pool = [1, 2, 3]
    subject_pool = [6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 27 ,28]
    del subject_pool[subject_pool.index(testing_subject)]

    X_train = pd.DataFrame()
    Y_train = pd.DataFrame()
    gp_train = pd.DataFrame()
    Y_test_result = []
    Y_pred_result = []

    for trial in trial_pool:
        for subject in subject_pool:
            for mode in ["RA2", "RA3", "RA4", "RA5", "RD2", "RD3", "RD4", "RD5", "SA1", "SA2", "SA3", "SA4", "SD1", "SD2", "SD3", "SD4"]:
                for starting_leg in ["R", "L"]:
                    train_path = fe_dir+"AB"+str(subject)+"_"+str(mode)+"_W"+str(window_size)+"_TP"+str(int(transition_point*100))+"_S2_"+str(starting_leg)+str(trial)+".csv"

                    if path.exists(train_path) == 1:
                        for train_read_path in glob.glob(train_path):
                            data = pd.read_csv(train_read_path, header=None)
                            X = data.iloc[:, feature_list]
                            Y = data.iloc[:, -1]
                            gp = data.iloc[:,-2]
                            X_train = pd.concat([X_train, X], axis=0, ignore_index=True)
                            Y_train = pd.concat([Y_train, Y], axis=0, ignore_index=True)
                            gp_train = pd.concat([gp_train, gp], axis=0, ignore_index=True)

            train_path = fe_dir+"AB"+str(subject)+"_LG_W"+str(window_size)+"_TP0_S2_R"+str(trial)+".csv"    
            if path.exists(train_path) == 1:
                for train_read_path in glob.glob(train_path):
                    data = pd.read_csv(train_read_path, header=None)
                    X = data.iloc[:, feature_list]
                    Y = data.iloc[:, -1]
                    gp = data.iloc[:,-2]
                    X_train = pd.concat([X_train, X], axis=0, ignore_index=True)
                    Y_train = pd.concat([Y_train, Y], axis=0, ignore_index=True)
                    gp_train = pd.concat([gp_train, gp], axis=0, ignore_index=True)

    svm_model = SVC(gamma='auto', kernel=kernel_type)
    svm_model.fit(X_train, np.ravel(Y_train))
    del [[X, Y, gp, X_train, Y_train, gp_train]]

    for mode in ["RA2", "RA3", "RA4", "RA5", "RD2", "RD3", "RD4", "RD5", "SA1", "SA2", "SA3", "SA4", "SD1", "SD2", "SD3", "SD4"]:
        for starting_leg in ["R", "L"]:   
            for trial in trial_pool: 
                test_path = fe_dir+"AB"+str(testing_subject)+"_"+str(mode)+"_W"+str(window_size)+"_TP"+str(int(transition_point*100))+"_S2_"+str(starting_leg)+str(trial)+".csv"
                if path.exists(test_path) == 1:
                    for test_read_path in glob.glob(test_path):
                        data = pd.read_csv(test_read_path, header=None)
                        X = data.iloc[:, feature_list]
                        Y = data.iloc[:, -1]
                        Y_pred = svm_model.predict(X)
                        Y_pred_result = np.concatenate((Y_pred_result, Y_pred))
                        Y_test_result = np.concatenate((Y_test_result, Y))

    for trial in trial_pool:
        train_path = fe_dir+"AB"+str(testing_subject)+"_LG_W"+str(window_size)+"_TP0_S2_R"+str(trial)+".csv"
        if path.exists(test_path) == 1:
            for test_read_path in glob.glob(test_path):
                data = pd.read_csv(test_read_path, header=None)
                X = data.iloc[:, feature_list]
                Y = data.iloc[:, -1]
                Y_pred = svm_model.predict(X)
                Y_pred_result = np.concatenate((Y_pred_result, Y_pred))
                Y_test_result = np.concatenate((Y_test_result, Y))
                del [[X, Y, Y_pred]]

    Y_test_result = np.ravel(Y_test_result)
    Y_pred_result = np.ravel(Y_pred_result)
    SVM_overall_accuracy = accuracy_score(Y_test_result, Y_pred_result)

    output = [channel, SVM_overall_accuracy]
    return output

###################################################################################
window_size = 350
transition_point = 0.2
phase_number = 1
kernel_type = 'rbf'
fe_dir = "/HDD/hipexo/Inseung/Feature Extraction Data/"

num_channels = 70
channel_list = list(np.arange(0,num_channels))
chosen_features = []
# # chosen_features = [1, 2, 52, 64, 12, 62, 54, 37, 44, 60, 53, 42, 43, 27, 32, 63, 11, 17, 24, 66, 38, 10, 45, 22, 7, 39, 51, 4, 21, 57, 14, 69, 34, 19, 25, 59, 29, 40, 15, 61, 18, 3, 41, 9, 36, 47, 33, 13, 8, 31, 6, 55, 48, 28, 49, 58, 20, 23, 16, 30, 68, 67, 46, 56, 65, 35, 50]
# # channel_list.remove(chosen_features)
# for x in chosen_features:
# 	channel_list.remove(x)
# print(channel_list)
# num_channels = len(channel_list)

for num_features in range(0, num_channels):
    print("Finding feature #" + str(num_features) + ': ')

    channel_scores = {}
    run_combos = []
    for channel in channel_list:
        for testing_subject in [6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 27 ,28]:
            run_combos.append([channel, testing_subject])
    joblib_result = Parallel(n_jobs=-1)(delayed(SVM_parallel)(combo) for combo in run_combos)

    for channel in channel_list:
        individual_channel_accuracy = []
        for model_output in joblib_result:
            if model_output[0] == channel:
                individual_channel_accuracy.append(model_output[1])
        channel_scores[channel] = np.mean(individual_channel_accuracy)

    #find max score
    highest_channel = max(channel_scores, key=channel_scores.get)
    print(str(highest_channel) + "\t" + str(channel_scores[highest_channel]))
    #append channel to list
    chosen_features.append(highest_channel)
    channel_list.remove(highest_channel)

    save_path = "/HDD/hipexo/Inseung/Result/SVM_FS_New.txt"
    with open(save_path, 'a') as f:
        for item in chosen_features:
            f.write(str(item) + ", ")
        f.write("\t" + str(channel_scores[highest_channel]) + "\n\r")
    f.close()


