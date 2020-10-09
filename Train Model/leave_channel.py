import glob
import os
import pandas as pd
import numpy as np
import math
import collections
import statistics
from os import path
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from joblib import Parallel, delayed
from sklearn import preprocessing
from sklearn.svm import SVC
import xgboost as xgb
import gc
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, LSTM, Conv1D, Flatten
from keras.optimizers import Adam, SGD, Nadam
from keras import backend as K
from keras import regularizers
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

fe_dir = "C:\\Users\\ikang7\\Desktop\\feature extraction data\\"
base_path_dir = "C:\\Users\\ikang7\\Desktop\\Result\\"

#############################################################################
def lda_parallel(combo):
    testing_subject = combo[0]
    window_size = combo[1]
    transition_point = combo[2]
    phase_number = combo[3]

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
        lda_model = LDA()
        lda_model.fit(X_train, np.ravel(Y_train))
        del [[X, Y, gp, X_train, Y_train, gp_train]]

        for mode in testing_mode:
            for starting_leg in ["R", "L"]:   
                for trial in trial_pool: 
                    test_path = fe_dir+"AB"+str(testing_subject)+"_"+str(mode)+"_W"+str(window_size)+"_TP"+str(int(transition_point*100))+"_S2_"+str(starting_leg)+str(trial)+".csv"

                    if path.exists(test_path) == 1:
                        for test_read_path in glob.glob(test_path):
                            data = pd.read_csv(test_read_path, header=None)
                            X = data.iloc[:, :-3]
                            for sensor in dropping_sensor:
                                X.iloc[:, sensor] = 0
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
                    for sensor in dropping_sensor:
                        X.iloc[:, sensor] = 0
                    Y = data.iloc[:, -1]
                    Y_pred = lda_model.predict(X)
                    Y_pred_result = np.concatenate((Y_pred_result, Y_pred))
                    Y_test_result = np.concatenate((Y_test_result, Y))
                    del [[X, Y, Y_pred]]

    else:
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
    text_file1 = base_path_dir + LDA_saving_file + ".txt"

    msg1 = ' '.join([str(testing_subject),str(window_size),str(transition_point),str(phase_number),str(LDA_overall_accuracy),"\n"])
    return text_file1, msg1
def SVM_parallel(combo):
    testing_subject = combo[0]
    window_size = combo[1]
    transition_point = combo[2]
    phase_number = combo[3]
    kernel_type = combo[4]

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
        svm_model = SVC(gamma='auto', kernel=kernel_type)
        svm_model.fit(X_train, np.ravel(Y_train))
        del [[X, Y, gp, X_train, Y_train, gp_train]]

        for mode in testing_mode:
            for starting_leg in ["R", "L"]:   
                for trial in trial_pool: 
                    test_path = fe_dir+"AB"+str(testing_subject)+"_"+str(mode)+"_W"+str(window_size)+"_TP"+str(int(transition_point*100))+"_S2_"+str(starting_leg)+str(trial)+".csv"

                    if path.exists(test_path) == 1:
                        for test_read_path in glob.glob(test_path):
                            data = pd.read_csv(test_read_path, header=None)
                            X = data.iloc[:, :-3]
                            for sensor in dropping_sensor:
                                X.iloc[:, sensor] = 0
                            Y = data.iloc[:, -1]
                            Y_pred = svm_model.predict(X)
                            Y_pred_result = np.concatenate((Y_pred_result, Y_pred))
                            Y_test_result = np.concatenate((Y_test_result, Y))

        for trial in trial_pool:
            train_path = fe_dir+"AB"+str(testing_subject)+"_LG_W"+str(window_size)+"_TP0_S2_R"+str(trial)+".csv"

            if path.exists(test_path) == 1:
                for test_read_path in glob.glob(test_path):
                    data = pd.read_csv(test_read_path, header=None)
                    X = data.iloc[:, :-3]
                    for sensor in dropping_sensor:
                        X.iloc[:, sensor] = 0
                    Y = data.iloc[:, -1]
                    Y_pred = svm_model.predict(X)
                    Y_pred_result = np.concatenate((Y_pred_result, Y_pred))
                    Y_test_result = np.concatenate((Y_test_result, Y))
                    del [[X, Y, Y_pred]]

    else:
        gp_train = gp_train.values
        gp_train[gp_train == 100] = 99.99
        gp_train_idx = []
        phase_model = []
        phase_count = np.arange(phase_number)

        for ii in phase_count:
            gp_train_idx.append([jj for jj, phase in enumerate(gp_train) if phase >= 0 + (ii/phase_number)*100 and phase < ((ii+1)/phase_number)*100])

        for ii in phase_count:
            svm_model = SVC(gamma='auto', kernel=kernel_type)
            svm_model.fit(X_train.values[gp_train_idx[ii]], np.ravel(Y_train.values[gp_train_idx[ii]]))          
            phase_model.append(svm_model)

        del [[X, Y, gp, X_train, Y_train, gp_train]]

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
    SVM_overall_accuracy = accuracy_score(Y_test_result, Y_pred_result)
    print("subject = "+str(testing_subject)+" window_size = "+str(window_size)+" phase_number = "+str(phase_number)+" Accuracy = "+str(SVM_overall_accuracy))
    text_file1 = base_path_dir + SVM_saving_file + ".txt"

    msg1 = ' '.join([str(testing_subject),str(window_size),str(transition_point),str(phase_number),str(SVM_overall_accuracy),"\n"])
    return text_file1, msg1
def NN_parallel(combo):
    testing_subject = combo[0]
    window_size = combo[1]
    transition_point = combo[2]
    phase_number = combo[3]
    layer_num = combo[4]
    node_num = combo[5]
    optimizer_value = combo[6]

    trial_pool = [1, 2, 3]
    subject_pool = [6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 27 ,28]
    del subject_pool[subject_pool.index(testing_subject)]

    X_train = pd.DataFrame()
    Y_train = pd.DataFrame()
    X_test = pd.DataFrame()
    Y_test = pd.DataFrame()
    gp_train = pd.DataFrame()
    gp_test = pd.DataFrame()
    Y_test_result = []
    Y_pred_result = []

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
        for mode in testing_mode:
            for starting_leg in ["R", "L"]:   
                for trial in trial_pool: 
                    test_path = fe_dir+"AB"+str(testing_subject)+"_"+str(mode)+"_W"+str(window_size)+"_TP"+str(int(transition_point*100))+"_S2_"+str(starting_leg)+str(trial)+".csv"

                    if path.exists(test_path) == 1:
                        for test_read_path in glob.glob(test_path):
                            data = pd.read_csv(test_read_path, header=None)
                            X = data.iloc[:, :-3]
                            for sensor in dropping_sensor:
                                X.iloc[:, sensor] = 0
                            Y = data.iloc[:, -1]
                            X_test = pd.concat([X_test, X], axis=0, ignore_index=True)
                            Y_test = pd.concat([Y_test, Y], axis=0, ignore_index=True)

        for trial in trial_pool:
            train_path = fe_dir+"AB"+str(testing_subject)+"_LG_W"+str(window_size)+"_TP0_S2_R"+str(trial)+".csv"

            if path.exists(test_path) == 1:
                for test_read_path in glob.glob(test_path):
                    data = pd.read_csv(test_read_path, header=None)
                    X = data.iloc[:, :-3]
                    for sensor in dropping_sensor:
                        X.iloc[:, sensor] = 0
                    Y = data.iloc[:, -1]
                    X_test = pd.concat([X_test, X], axis=0, ignore_index=True)
                    Y_test = pd.concat([Y_test, Y], axis=0, ignore_index=True)
        del [[gp, gp_train, X, Y]]

        NN_model = Sequential()
        if layer_num == 1:
            NN_model.add(Dense(node_num, activation='relu', input_shape=(X_train.shape[1],)))
            NN_model.add(Dense(5, activation='softmax'))
        else:
            ii = 0
            NN_model.add(Dense(node_num, activation='relu', input_shape=(X_train.shape[1],)))
            while ii < layer_num-1:
                ii = ii + 1
                NN_model.add(Dense(node_num, activation='relu'))
            NN_model.add(Dense(5, activation='softmax'))
        NN_model.compile(optimizer=optimizer_value, loss='sparse_categorical_crossentropy', metrics=['acc'])    
        h = NN_model.fit(X_train, np.ravel(Y_train), epochs=200, batch_size=128, verbose=0, validation_data=(X_test, Y_test),shuffle=True,callbacks=[EarlyStopping(patience=100,restore_best_weights=True)])
        idx = np.argmin(h.history['val_loss'])
        NN_overall_accuracy = h.history['val_acc'][idx]
        del [[X_train, Y_train, X_test, Y_test]]

    else:
        for mode in testing_mode:
            for starting_leg in ["R", "L"]:   
                for trial in trial_pool: 
                    test_path = fe_dir+"AB"+str(testing_subject)+"_"+str(mode)+"_W"+str(window_size)+"_TP"+str(int(transition_point*100))+"_S2_"+str(starting_leg)+str(trial)+".csv"

                    if path.exists(test_path) == 1:
                        for test_read_path in glob.glob(test_path):
                            data = pd.read_csv(test_read_path, header=None)
                            X = data.iloc[:, :-3]
                            Y = data.iloc[:, -1]
                            gp = data.iloc[:,-2]
                            X_test = pd.concat([X_test, X], axis=0, ignore_index=True)
                            Y_test = pd.concat([Y_test, Y], axis=0, ignore_index=True)
                            gp_test = pd.concat([gp_test, gp], axis=0, ignore_index=True)

        for trial in trial_pool:
            train_path = fe_dir+"AB"+str(testing_subject)+"_LG_W"+str(window_size)+"_TP0_S2_R"+str(trial)+".csv"

            if path.exists(test_path) == 1:
                for test_read_path in glob.glob(test_path):
                    data = pd.read_csv(test_read_path, header=None)
                    X = data.iloc[:, :-3]
                    Y = data.iloc[:, -1]
                    gp = data.iloc[:,-2]
                    X_test = pd.concat([X_test, X], axis=0, ignore_index=True)
                    Y_test = pd.concat([Y_test, Y], axis=0, ignore_index=True)
                    gp_test = pd.concat([gp_test, gp], axis=0, ignore_index=True)
        del [[gp, X, Y]]

        gp_train = gp_train.values
        gp_test = gp_test.values
        gp_train[gp_train == 100] = 99.99
        gp_test[gp_test == 100] = 99.99
        gp_train_idx = []
        gp_test_idx = []
        phase_model = []
        phase_count = np.arange(phase_number)

        for ii in phase_count:
            gp_train_idx.append([jj for jj, phase in enumerate(gp_train) if phase >= 0 + (ii/phase_number)*100 and phase < ((ii+1)/phase_number)*100])
            gp_test_idx.append([jj for jj, phase in enumerate(gp_test) if phase >= 0 + (ii/phase_number)*100 and phase < ((ii+1)/phase_number)*100])

        for ii in phase_count:
            NN_model = Sequential()
            if layer_num == 1:
                NN_model.add(Dense(node_num, activation='relu', input_shape=(X_train.shape[1],)))
                NN_model.add(Dense(5, activation='softmax'))
            else:
                counter = 0
                NN_model.add(Dense(node_num, activation='relu', input_shape=(X_train.shape[1],)))
                while counter < layer_num-1:
                    counter = counter + 1
                    NN_model.add(Dense(node_num, activation='relu'))
                NN_model.add(Dense(5, activation='softmax'))

            NN_model.compile(optimizer=optimizer_value, loss='sparse_categorical_crossentropy', metrics=['acc'])
            NN_model.fit(X_train.values[gp_train_idx[ii]], np.ravel(Y_train.values[gp_train_idx[ii]]), epochs=200, batch_size=128, verbose=0,callbacks=[EarlyStopping(patience=100,restore_best_weights=True)])
            phase_model.append(NN_model)

        for ii in phase_count:
            Y_pred = phase_model[ii].predict_classes(X_test.values[gp_test_idx[ii]])
            Y_pred_result.append(Y_pred)
            Y_test_result.append(np.ravel(Y_test.values[gp_test_idx[ii]]))

        del [[X_train, Y_train, X_test, Y_test, gp_train, gp_test]]

        Y_pred_result = np.concatenate(Y_pred_result).ravel()
        Y_test_result = np.concatenate(Y_test_result).ravel()
        NN_overall_accuracy = accuracy_score(Y_test_result, Y_pred_result)

    print("subject = "+str(testing_subject)+" window size = "+str(window_size)+ " phase number = "+ str(phase_number)+" Accuracy = "+str(NN_overall_accuracy))
    
    text_file1 = base_path_dir + NN_saving_file + ".txt"
    msg1 = ' '.join([str(testing_subject),str(window_size),str(transition_point),str(phase_number),str(NN_overall_accuracy),"\n"])
    return text_file1, msg1
def xgboost_parallel(combo):
    testing_subject = combo[0]
    window_size = combo[1]
    transition_point = combo[2]
    phase_number = combo[3]
    boost_round = combo[4]
    tree_depth = combo[5]
    child_weight = combo[6]

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
        xg_train = xgb.DMatrix(X_train, label=Y_train)
        model = xgb.train(params, xg_train, num_boost_round = boost_round)
        del [[X, Y, gp, X_train, Y_train, gp_train]]

        for mode in testing_mode:
            for starting_leg in ["R", "L"]:   
                for trial in trial_pool: 
                    test_path = fe_dir+"AB"+str(testing_subject)+"_"+str(mode)+"_W"+str(window_size)+"_TP"+str(int(transition_point*100))+"_S2_"+str(starting_leg)+str(trial)+".csv"

                    if path.exists(test_path) == 1:
                        for test_read_path in glob.glob(test_path):
                            data = pd.read_csv(test_read_path, header=None)
                            X = data.iloc[:, :-3]
                            for sensor in dropping_sensor:
                                X.iloc[:, sensor] = 0
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
                    X = data.iloc[:, :-3]
                    for sensor in dropping_sensor:
                        X.iloc[:, sensor] = 0
                    Y = data.iloc[:, -1]
                    xg_test = xgb.DMatrix(X, label=Y)
                    Y_pred = model.predict(xg_test)

                    Y_pred_result = np.concatenate((Y_pred_result, Y_pred))
                    Y_test_result = np.concatenate((Y_test_result, Y))
                    del [[X, Y, Y_pred, xg_test]]

    else:
        gp_train = gp_train.values
        gp_train[gp_train == 100] = 99.99
        gp_train_idx = []
        phase_model = []
        phase_count = np.arange(phase_number)

        for ii in phase_count:
            gp_train_idx.append([jj for jj, phase in enumerate(gp_train) if phase >= 0 + (ii/phase_number)*100 and phase < ((ii+1)/phase_number)*100])

        for ii in phase_count:
            xg_train = xgb.DMatrix(X_train.values[gp_train_idx[ii]], label=Y_train.values[gp_train_idx[ii]])
            model = xgb.train(params, xg_train, num_boost_round = boost_round)      
            phase_model.append(model)
        del [[X, Y, gp, X_train, Y_train, gp_train, xg_train]]

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
                                        xg_test = xgb.DMatrix(X.values[ii,:].reshape(1, -1))
                                        Y_pred = phase_model[jj].predict(xg_test)
                                        Y_pred_result.append(Y_pred)
                            Y_test_result = np.concatenate((Y_test_result, Y))
                            del [[X, Y, gp, Y_pred, xg_test]]

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
                                xg_test = xgb.DMatrix(X.values[ii,:].reshape(1, -1))
                                Y_pred = phase_model[jj].predict(xg_test)
                                Y_pred_result.append(Y_pred)
                    Y_test_result = np.concatenate((Y_test_result, Y))
                    del [[X, Y, gp, Y_pred, xg_test]]

    Y_test_result = np.ravel(Y_test_result)
    Y_pred_result = np.ravel(Y_pred_result)

    xgboost_overall_accuracy = accuracy_score(Y_test_result, Y_pred_result)
    print("subject = "+str(testing_subject)+" window_size = "+str(window_size)+" phase_number = "+str(phase_number)+" Accuracy = "+str(xgboost_overall_accuracy))

    text_file1 = base_path_dir + XGB_saving_file + ".txt"

    msg1 = ' '.join([str(testing_subject),str(window_size),str(transition_point),str(phase_number),str(xgboost_overall_accuracy),"\n"])
    return text_file1, msg1

# hip 0 - 9 channel
# thigh 10 - 39 channel
# trunk 40 - 69 channel

training_mode = ["RA2", "RA3", "RA4", "RA5", "RD2", "RD3", "RD4", "RD5", "SA1", "SA2", "SA3", "SA4", "SD1", "SD2", "SD3", "SD4"]
testing_mode = ["RA2", "RA3", "RA4", "RA5", "RD2", "RD3", "RD4", "RD5", "SA1", "SA2", "SA3", "SA4", "SD1", "SD2", "SD3", "SD4"]

for channel in np.arange(13, 14):
    dropping_sensor = np.arange(channel*5, channel*5+4)

    LDA_saving_file = "LDA_channel_remove" + str(channel)
    SVM_saving_file = "SVM_channel_remove" + str(channel)
    NN_saving_file = "NN_channel_remove" + str(channel)
    XGB_saving_file = "XGB_channel_remove" + str(channel)

    # LDA run
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

    # SVM run
    run_combos = []
    for testing_subject in [6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 27 ,28]:
        for window_size in [350]:
            for transition_point in [0.2]:
                for phase_number in [1]:
                    for kernel_type in ['rbf']:
                            run_combos.append([testing_subject, window_size, transition_point, phase_number, kernel_type])
    result = Parallel(n_jobs=-1)(delayed(SVM_parallel)(combo) for combo in run_combos)
    for r in result:
        with open(r[0],"a+") as f:
            f.write(r[1])

    # NN run
    run_combos = []
    for testing_subject in [6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 27 ,28]:
        for window_size in [550]:
                for phase_number in [1]:
                    for layer_num in [4]:
                        for node_num in [25]:
                            for optimizer_value in ['SGD']:
                                run_combos.append([testing_subject, window_size, transition_point, phase_number, layer_num, node_num, optimizer_value])
    result = Parallel(n_jobs=-1)(delayed(NN_parallel)(combo) for combo in run_combos)
    for r in result:
        with open(r[0],"a+") as f:
            f.write(r[1])

    # XGB run
    run_combos = []
    for testing_subject in [6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 27 ,28]:
        for window_size in [350]:
            for transition_point in [0.2]:
                for phase_number in [1]:
                    for boost_round in [200]:
                        for tree_depth in [8]:
                            for child_weight in [0.01]:
                                run_combos.append([testing_subject, window_size, transition_point, phase_number, boost_round, tree_depth, child_weight])
    result = Parallel(n_jobs=-1)(delayed(xgboost_parallel)(combo) for combo in run_combos)
    for r in result:
        with open(r[0],"a+") as f:
            f.write(r[1])

