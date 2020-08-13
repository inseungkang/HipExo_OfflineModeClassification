import pandas as pd
import numpy as np
import math
from sklearn import preprocessing
import matplotlib.pyplot as plt
from matplotlib import style
import glob
import os
from sklearn.model_selection import train_test_split, cross_val_score, KFold
import collections
import statistics
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.ensemble import AdaBoostClassifier as AdaBoost
from sklearn.ensemble import RandomForestClassifier
from xgboost.sklearn import XGBClassifier as xgb
from sklearn.svm import LinearSVC
from sklearn import neighbors
from joblib import Parallel, delayed

os.environ["CUDA_VISIBLE_DEVICES"] = '-1'

def get_data(subject, trial_number, window_size, data_type):
    base_path_dir = "/HDD/hipexo/Dropbox (GaTech)/Mode Classification/Data/Inseung/"
    fe_dir = base_path_dir + "Feature Data/"

    X_train = pd.DataFrame()
    Y_train = pd.DataFrame()
    X_test = pd.DataFrame()
    Y_test = pd.DataFrame()

    if trial_number in [1, 2, 3, 4]:
        train_trial = [1, 2, 3, 4]
        test_trial = trial_number
        del train_trial[test_trial-1]
    elif trial_number in [5, 6, 7, 8]:
        train_trial = [5, 6, 7, 8]
        test_trial = trial_number
        del train_trial[test_trial-5]

    for mode in ["RA1", "RA2", "RA3", "RA4", "RD1", "RD2", "RD3", "RD4", "SD1", "SD2", "SD3", "SD4", "SA1", "SA2", "SA3", "SA4", "L0"]:
        for starting_leg in ["R", "L"]:
            if data_type == "MECH":

                for i in range(len(train_trial)):
                    train_file_path = fe_dir+"AB0"+str(subject)+"/AB0"+str(subject)+"_"+str(mode)+"_W"+str(window_size*10)+"_"+str(starting_leg)+str(train_trial[i])+"_"+str(data_type)+".csv"

                    for train_read_path in glob.glob(train_file_path):
                        data = pd.read_csv(train_read_path, header=None)
                        X = data.iloc[:, :-3]
                        Y = data.iloc[:, -3:]
                        X_train = pd.concat([X_train, X], axis=0, ignore_index=True)
                        Y_train = pd.concat([Y_train, Y], axis=0, ignore_index=True)

            elif data_type == "COMBINED":
                for i in range(len(train_trial)):
                    train_file_path = fe_dir+"AB0"+str(subject)+"/AB0"+str(subject)+"_"+str(mode)+"_W"+str(window_size*10)+"_"+str(starting_leg)+str(train_trial[i])+"_"+str(data_type)+".csv"    

                    for train_read_path in glob.glob(train_file_path):
                        data = pd.read_csv(train_read_path, header=None)
                        X = data.iloc[:, :-3]
                        Y = data.iloc[:, -3:]
                        X_train = pd.concat([X_train, X], axis=0, ignore_index=True)
                        Y_train = pd.concat([Y_train, Y], axis=0, ignore_index=True)


    return X_train, Y_train

def get_test_data(subject, trial_number, window_size, data_type, mode, starting_leg):
    base_path_dir = "/HDD/hipexo/Dropbox (GaTech)/Mode Classification/Data/Inseung/"
    fe_dir = base_path_dir + "Feature Data/"

    X_test = pd.DataFrame()
    Y_test = pd.DataFrame()

    test_file_path = fe_dir+"AB0"+str(subject)+"/AB0"+str(subject)+"_"+str(mode)+"_W"+str(window_size*10)+"_"+str(starting_leg)+str(trial_number)+"_"+str(data_type)+".csv"

    for test_read_path in glob.glob(test_file_path):
        data = pd.read_csv(test_read_path, header=None)
        X = data.iloc[:, :-3]
        Y = data.iloc[:, -3:]
        X_test = pd.concat([X_test, X], axis=0, ignore_index=True)
        Y_test = pd.concat([Y_test, Y], axis=0, ignore_index=True)

    return X_test, Y_test


def speed_feature_selection(combo):
    subject = combo[0]

    window_size = 60
    phase_number = 5
    data_type = "COMBINED"
    save_path = "/HDD/hipexo/Dropbox (GaTech)/Mode Classification/Data/Inseung/Result/feature_selection.csv"

    num_channels = 70 # = num_features =78
    channel_list = list( np.arange(0, num_channels) ) # len(channel_list) = 78

    chosen_features = []
    for num_features in range(0, num_channels):

        print("Finding feature #" + str(num_features) + ': ')
        channel_scores = {}

        for channel in channel_list:
            testing_features = chosen_features + [channel]
            feature_list = testing_features
            for trial_number in [1, 2, 3, 4, 5, 6, 7, 8]:
                X_train, Y_train = get_data(subject, trial_number, window_size, data_type)
                X_train = X_train.iloc[:, feature_list]
                Y_train = Y_train.iloc[:,:]

                X_train = X_train.values
                Y_train = Y_train.values

                ave_acc = []
                Y_test_result = []
                Y_pred_result = []
                Y_test_mode = []
                for mode in ["RA1", "RA2", "RA3", "RA4", "RD1", "RD2", "RD3", "RD4", "SD1", "SD2", "SD3", "SD4", "SA1", "SA2", "SA3", "SA4"]:
                    for starting_leg in ["R", "L"]:
                        print("subject = "+ str(subject) + ", mode = " + mode + ", leg = " + starting_leg + ", phase number = " + str(phase_number) + ", trial = " + str(trial_number))
                        X_test, Y_test = get_test_data(subject, trial_number, window_size, data_type, mode, starting_leg)
                        X_test = X_test.iloc[:, feature_list]
                        X_test = X_test.values
                        Y_test = Y_test.values

                        Y_pred_trans = []
                        Y_train_temp = pd.DataFrame(Y_train)
                        Y_test = pd.DataFrame(Y_test)

                        train_gait_phase = Y_train_temp.iloc[:, -3].reset_index(drop=True)
                        test_gait_phase = Y_test.iloc[:, -3].reset_index(drop=True)
                        Y_test_transition = Y_test.iloc[:, -2].reset_index(drop=True)
                        Y_test = Y_test.iloc[:, -1].reset_index(drop=True)
                        Y_train_temp = Y_train[:,-1]
                        Y_test = Y_test.values

                        train_phase_idx_bin = []
                        phase_count = np.arange(phase_number)

                        for i in phase_count:
                            idx = [j for j, phase in enumerate(train_gait_phase.values) if phase >= 0.0 + i/phase_number and phase < (i+1)/phase_number]
                            train_phase_idx_bin.append(idx)

                        phase_LDA_model = []
                        for i in phase_count:
                            LDA_model = QDA()
                            LDA_model.fit(X_train[train_phase_idx_bin[i]], np.ravel(Y_train_temp[train_phase_idx_bin[i]]))
                            phase_LDA_model.append(LDA_model)

                        for i in range(len(Y_test)):
                            for j in phase_count:
                                if test_gait_phase[i] >= 0 + j/phase_number and test_gait_phase[i] < (j+1)/phase_number:
                                    Y_pred = phase_LDA_model[j].predict(X_test[i,:].reshape(1, -1))
                                    Y_pred_result.append(Y_pred)

                        Y_test_result = np.concatenate((Y_test_result, Y_test))
                        Y_test_mode = np.concatenate((Y_test_mode, Y_test_transition))

                trans_idx = np.argwhere(Y_test_mode == 5)
                steady_idx = np.argwhere(Y_test_mode != 5)
                Y_test_result = np.ravel(Y_test_result)
                Y_pred_result = np.ravel(Y_pred_result)

                LDA_steady_accuracy = accuracy_score(Y_test_result[steady_idx], Y_pred_result[steady_idx])
                ave_acc.append(LDA_steady_accuracy)

            score = np.average(ave_acc)
            print(score)
            channel_scores[channel] = score

        #find max score
        highest_channel = min(channel_scores, key=channel_scores.get)
        print(str(highest_channel) + "\t" + str(channel_scores[highest_channel]))
        #append channel to list
        chosen_features.append(highest_channel)
        channel_list.remove(highest_channel)

        base_path_dir = "/HDD/hipexo/Dropbox (GaTech)/Mode Classification/Data/Inseung/"
        text_file1 = base_path_dir + "Result/feature_selection.txt"
        msg1 = ' '.join([str(chosen_features), str(channel_scores[highest_channel]), "\n"])

    return text_file1, msg1


run_combos = []
for subject in [2]:
    run_combos.append([subject])

result = Parallel(n_jobs=-1)(delayed(speed_feature_selection)(combo) for combo in run_combos)

for r in result:
    with open(r[0],"a+") as f:
        f.write(r[1])