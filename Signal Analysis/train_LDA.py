import glob
import os
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import style
import collections
import statistics
from os import path
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from joblib import Parallel, delayed


LDA_saving_file = "LDA_remove2"
SVM_saving_file = "SVM_remove2"
NN_saving_file = "NN_remove2"
XGB_saving_file = "XGB_remove2"

training_mode = ["RA2", "RA4", "RA5", "RD2", "RD4", "RD5","SA1", "SA3", "SA4", "SD1", "SD3", "SD4"]
testing_mode = ["RA3", "RD3", "SA2", "SD2"]


#############################################################################
def lda_parallel(combo):
    testing_subject = combo[0]
    window_size = combo[1]
    transition_point = combo[2]
    phase_number = combo[3]

    fe_dir = "/HDD/hipexo/Inseung/feature extraction data/"

    trial_pool = [1, 2, 3]
    subject_pool = [6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 27 ,28]
    del subject_pool[subject_pool.index(testing_subject)]

    X_train = pd.DataFrame()
    Y_train = pd.DataFrame()
    gp_train = pd.DataFrame()
    Y_test_result = []
    Y_pred_result = []

######### concat all the training data ##############
    for trial in trial_pool:
        for subject in subject_pool:
            for mode in training_mode:
                for starting_leg in ["R", "L"]:
                    train_path = fe_dir+"AB"+str(subject)+"_"+str(mode)+"_W"+str(window_size)+"_TP"+str(int(transition_point*100))+"_S2_"+str(starting_leg)+str(trial)+".csv"

                    if path.exists(train_path) == 1:
                        for train_read_path in glob.glob(train_path):
                            data = pd.read_csv(train_read_path, header=None)
                            X = data.iloc[:, :-3]
                            Y = data.iloc[:, -1]
                            gp = data.iloc[:,-2]
                            X_train = pd.concat([X_train, X], axis=0, ignore_index=True)
                            Y_train = pd.concat([Y_train, Y], axis=0, ignore_index=True)
                            gp_train = pd.concat([gp_train, gp], axis=0, ignore_index=True)

            train_path = fe_dir+"AB"+str(subject)+"_LG_W"+str(window_size)+"_TP0_S2_R"+str(trial)+".csv"    
            if path.exists(train_path) == 1:
                for train_read_path in glob.glob(train_path):
                    data = pd.read_csv(train_read_path, header=None)
                    X = data.iloc[:, :-3]
                    Y = data.iloc[:, -1]
                    gp = data.iloc[:,-2]
                    X_train = pd.concat([X_train, X], axis=0, ignore_index=True)
                    Y_train = pd.concat([Y_train, Y], axis=0, ignore_index=True)
                    gp_train = pd.concat([gp_train, gp], axis=0, ignore_index=True)

    if phase_number == 1:
######### training the unified model ##############
        lda_model = LDA()
        lda_model.fit(X_train, np.ravel(Y_train))
        del [[X, Y, gp, X_train, Y_train, gp_train]]

######### testing the unified model ##############
        for mode in testing_mode:
            for starting_leg in ["R", "L"]:   
                for trial in trial_pool: 
                    test_path = fe_dir+"AB"+str(testing_subject)+"_"+str(mode)+"_W"+str(window_size)+"_TP"+str(int(transition_point*100))+"_S2_"+str(starting_leg)+str(trial)+".csv"

                    if path.exists(test_path) == 1:
                        for test_read_path in glob.glob(test_path):
                            data = pd.read_csv(test_read_path, header=None)
                            X = data.iloc[:, :-3]
                            Y = data.iloc[:, -1]
                            Y_pred = lda_model.predict(X)
                            Y_pred_result = np.concatenate((Y_pred_result, Y_pred))
                            Y_test_result = np.concatenate((Y_test_result, Y))

        for trial in trial_pool:
            train_path = fe_dir+"AB"+str(testing_subject)+"_LG_W"+str(window_size)+"_TP0_S2_R"+str(trial)+".csv"

            if path.exists(test_path) == 1:
                for test_read_path in glob.glob(test_path):
                    data = pd.read_csv(test_read_path, header=None)
                    X = data.iloc[:, :-3]
                    Y = data.iloc[:, -1]
                    Y_pred = lda_model.predict(X)
                    Y_pred_result = np.concatenate((Y_pred_result, Y_pred))
                    Y_test_result = np.concatenate((Y_test_result, Y))
                    del [[X, Y, Y_pred]]

    else:
######### training the phase dependent model ##############
        gp_train = gp_train.values
        gp_train[gp_train == 100] = 99.99
        gp_train_idx = []
        phase_model = []
        phase_count = np.arange(phase_number)

        for ii in phase_count:
            gp_train_idx.append([jj for jj, phase in enumerate(gp_train) if phase >= 0 + (ii/phase_number)*100 and phase < ((ii+1)/phase_number)*100])

        for ii in phase_count:
            lda_model = LDA()
            lda_model.fit(X_train.values[gp_train_idx[ii]], np.ravel(Y_train.values[gp_train_idx[ii]]))
            phase_model.append(lda_model)

        del [[X, Y, gp, X_train, Y_train, gp_train]]

######### testing the phase dependent model ##############
        for mode in testing_mode:
            for starting_leg in ["R", "L"]:   
                for trial in trial_pool: 
                    test_path = fe_dir+"AB"+str(testing_subject)+"_"+str(mode)+"_W"+str(window_size)+"_TP"+str(int(transition_point*100))+"_S2_"+str(starting_leg)+str(trial)+".csv"

                    if path.exists(test_path) == 1:
                        for test_read_path in glob.glob(test_path):
                            data = pd.read_csv(test_read_path, header=None)
                            X = data.iloc[:, :-3]
                            Y = data.iloc[:, -1]
                            gp = data.iloc[:, -2].values
                            gp[gp == 100] = 99.99

                            for ii in range(len(Y)):
                                for jj in phase_count:
                                    if gp[ii] >= 0 + (jj/phase_number)*100 and gp[ii] < ((jj+1)/phase_number)*100:
                                        Y_pred = phase_model[jj].predict(X.values[ii,:].reshape(1, -1))
                                        Y_pred_result.append(Y_pred)                            
                            Y_test_result = np.concatenate((Y_test_result, Y))


        for trial in trial_pool:
            train_path = fe_dir+"AB"+str(testing_subject)+"_LG_W"+str(window_size)+"_TP0_S2_R"+str(trial)+".csv"

            if path.exists(test_path) == 1:
                for test_read_path in glob.glob(test_path):
                    data = pd.read_csv(test_read_path, header=None)
                    X = data.iloc[:, :-3]
                    Y = data.iloc[:, -1]
                    gp = data.iloc[:, -2].values
                    gp[gp == 100] = 99.99

                    for ii in range(len(Y)):
                        for jj in phase_count:
                            if gp[ii] >= 0 + (jj/phase_number)*100 and gp[ii] < ((jj+1)/phase_number)*100:
                                Y_pred = phase_model[jj].predict(X.values[ii,:].reshape(1, -1))
                                Y_pred_result.append(Y_pred)                            
                    Y_test_result = np.concatenate((Y_test_result, Y))
                    del [[X, Y, gp, Y_pred]]

    Y_test_result = np.ravel(Y_test_result)
    Y_pred_result = np.ravel(Y_pred_result)
    LDA_overall_accuracy = accuracy_score(Y_test_result, Y_pred_result)
    print("subject = "+str(testing_subject)+" window size = "+str(window_size)+" phase number = "+str(phase_number)+ " Accuracy = "+str(LDA_overall_accuracy))

    base_path_dir = "/HDD/hipexo/Inseung/Result/"
    text_file1 = base_path_dir + LDA_saving_file + ".txt"

    msg1 = ' '.join([str(testing_subject), str(window_size), str(transition_point), str(phase_number), str(LDA_overall_accuracy),"\n"])
    return text_file1, msg1


run_combos = []
for testing_subject in [6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 27 ,28]:
    for window_size in [750]:
        for transition_point in [0.2]:
            for phase_number in [1]:
                run_combos.append([testing_subject, window_size, transition_point, phase_number])
result = Parallel(n_jobs=-1)(delayed(lda_parallel)(combo) for combo in run_combos)
for r in result:
    with open(r[0],"a+") as f:
        f.write(r[1])