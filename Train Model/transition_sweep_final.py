import glob
import os
import gc
import pandas as pd
import numpy as np
import math
import collections
import statistics
from os import path
from joblib import Parallel, delayed
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, LSTM, Conv1D, Flatten
from keras.optimizers import Adam, SGD, Nadam
from keras import backend as K
from keras import regularizers
from sklearn.model_selection import train_test_split, cross_val_score, KFold
import tensorflow as tf
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
np.random.seed(1)
tf.set_random_seed(seed=5)

LDA_saving_file = "LDA_transition_sweep"
SVM_saving_file = "SVM_transition_sweep"
NN_saving_file = "NN_transition_sweep"
XGB_saving_file = "XGB_transition_sweep"

training_mode = ["RA2", "RA3", "RA4", "RA5", "RD2", "RD3", "RD4", "RD5", "SA1", "SA2", "SA3", "SA4", "SD1", "SD2", "SD3", "SD4"]
testing_mode = ["RA2", "RA3", "RA4", "RA5", "RD2", "RD3", "RD4", "RD5", "SA1", "SA2", "SA3", "SA4", "SD1", "SD2", "SD3", "SD4"]

fe_dir = "/HDD/hipexo/Inseung/feature extraction data/"
base_path_dir = "/HDD/hipexo/Inseung/Result/"

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

    Y_steady_pred_result = []
    Y_steady_test_result = []
    Y_trans_pred_result = []
    Y_trans_test_result = []

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
                        Y = data.iloc[:, -1]

                        Y_True = data.iloc[:, -3]
                        trans_idx = [i for i, x in enumerate(Y_True) if x == 5]
                        steady_idx = list(set(np.arange(0,len(Y))) - set(trans_idx))

                        Y_pred = lda_model.predict(X)                        
                        Y_pred_result = np.concatenate((Y_pred_result, Y_pred))
                        Y_test_result = np.concatenate((Y_test_result, Y))
                        Y_steady_pred_result = np.concatenate((Y_steady_pred_result, Y_pred[steady_idx]))
                        Y_steady_test_result = np.concatenate((Y_steady_test_result, Y[steady_idx])) 
                        Y_trans_pred_result = np.concatenate((Y_trans_pred_result, Y_pred[trans_idx])) 
                        Y_trans_test_result = np.concatenate((Y_trans_test_result, Y[trans_idx])) 

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
                Y_steady_pred_result = np.concatenate((Y_steady_pred_result, Y_pred))
                Y_steady_test_result = np.concatenate((Y_steady_test_result, Y))
                del [[X, Y, Y_pred]]

    Y_test_result = np.ravel(Y_test_result)
    Y_pred_result = np.ravel(Y_pred_result)
    Y_steady_test_result = np.ravel(Y_steady_test_result)
    Y_steady_pred_result = np.ravel(Y_steady_pred_result)
    Y_trans_test_result = np.ravel(Y_trans_test_result)
    Y_trans_pred_result = np.ravel(Y_trans_pred_result)

    LDA_overall_accuracy = accuracy_score(Y_test_result, Y_pred_result) 
    LDA_steady_accuracy = accuracy_score(Y_steady_test_result, Y_steady_pred_result)
    LDA_trans_accuracy = accuracy_score(Y_trans_test_result, Y_trans_pred_result)

    text_file1 = base_path_dir + LDA_saving_file + ".txt"
    msg1 = ' '.join([str(testing_subject),str(window_size),str(transition_point),str(phase_number),str(LDA_steady_accuracy),str(LDA_trans_accuracy),str(LDA_overall_accuracy),"\n"])
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

    Y_steady_pred_result = []
    Y_steady_test_result = []
    Y_trans_pred_result = []
    Y_trans_test_result = []

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
                        Y = data.iloc[:, -1]

                        Y_True = data.iloc[:, -3]
                        trans_idx = [i for i, x in enumerate(Y_True) if x == 5]
                        steady_idx = list(set(np.arange(0,len(Y))) - set(trans_idx))

                        Y_pred = svm_model.predict(X)                     
                        Y_pred_result = np.concatenate((Y_pred_result, Y_pred))
                        Y_test_result = np.concatenate((Y_test_result, Y))
                        Y_steady_pred_result = np.concatenate((Y_steady_pred_result, Y_pred[steady_idx]))
                        Y_steady_test_result = np.concatenate((Y_steady_test_result, Y[steady_idx])) 
                        Y_trans_pred_result = np.concatenate((Y_trans_pred_result, Y_pred[trans_idx])) 
                        Y_trans_test_result = np.concatenate((Y_trans_test_result, Y[trans_idx])) 

    for trial in trial_pool:
        train_path = fe_dir+"AB"+str(testing_subject)+"_LG_W"+str(window_size)+"_TP0_S2_R"+str(trial)+".csv"

        if path.exists(test_path) == 1:
            for test_read_path in glob.glob(test_path):
                data = pd.read_csv(test_read_path, header=None)
                X = data.iloc[:, :-3]
                Y = data.iloc[:, -1]
                Y_pred = svm_model.predict(X)
                Y_pred_result = np.concatenate((Y_pred_result, Y_pred))
                Y_test_result = np.concatenate((Y_test_result, Y))
                Y_steady_pred_result = np.concatenate((Y_steady_pred_result, Y_pred))
                Y_steady_test_result = np.concatenate((Y_steady_test_result, Y))
                del [[X, Y, Y_pred]]

    Y_test_result = np.ravel(Y_test_result)
    Y_pred_result = np.ravel(Y_pred_result)

    Y_steady_test_result = np.ravel(Y_steady_test_result)
    Y_steady_pred_result = np.ravel(Y_steady_pred_result)

    Y_trans_test_result = np.ravel(Y_trans_test_result)
    Y_trans_pred_result = np.ravel(Y_trans_pred_result)

    SVM_overall_accuracy = accuracy_score(Y_test_result, Y_pred_result) 
    SVM_steady_accuracy = accuracy_score(Y_steady_test_result, Y_steady_pred_result)
    SVM_trans_accuracy = accuracy_score(Y_trans_test_result, Y_trans_pred_result)

    text_file1 = base_path_dir + SVM_saving_file + ".txt"
    msg1 = ' '.join([str(testing_subject),str(window_size),str(transition_point),str(phase_number),str(SVM_steady_accuracy),str(SVM_trans_accuracy),str(SVM_overall_accuracy),"\n"])
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
    Y_True_data = pd.DataFrame()
    gp_train = pd.DataFrame()
    gp_test = pd.DataFrame()

    Y_steady_pred_result = []
    Y_steady_test_result = []
    Y_trans_pred_result = []
    Y_trans_test_result = []

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


    for mode in testing_mode:
        for starting_leg in ["R", "L"]:   
            for trial in trial_pool: 
                test_path = fe_dir+"AB"+str(testing_subject)+"_"+str(mode)+"_W"+str(window_size)+"_TP"+str(int(transition_point*100))+"_S2_"+str(starting_leg)+str(trial)+".csv"

                if path.exists(test_path) == 1:
                    for test_read_path in glob.glob(test_path):
                        data = pd.read_csv(test_read_path, header=None)
                        X = data.iloc[:, :-3]
                        Y = data.iloc[:, -1]
                        X_test = pd.concat([X_test, X], axis=0, ignore_index=True)
                        Y_test = pd.concat([Y_test, Y], axis=0, ignore_index=True)

                        Y_True = data.iloc[:, -3]
                        Y_True_data = pd.concat([Y_True_data, Y_True], axis=0, ignore_index=True)

    for trial in trial_pool:
        train_path = fe_dir+"AB"+str(testing_subject)+"_LG_W"+str(window_size)+"_TP0_S2_R"+str(trial)+".csv"

        if path.exists(test_path) == 1:
            for test_read_path in glob.glob(test_path):
                data = pd.read_csv(test_read_path, header=None)
                X = data.iloc[:, :-3]
                Y = data.iloc[:, -1]
                X_test = pd.concat([X_test, X], axis=0, ignore_index=True)
                Y_test = pd.concat([Y_test, Y], axis=0, ignore_index=True)
                Y_True = data.iloc[:, -3]
                Y_True_data = pd.concat([Y_True_data, Y_True], axis=0, ignore_index=True)

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
    NN_model.fit(X_train, np.ravel(Y_train), epochs=200, batch_size=128, verbose=0, validation_data=(X_test, Y_test),shuffle=True,callbacks=[EarlyStopping(patience=100,restore_best_weights=True)])
    
    trans_idx = [i for i, x in enumerate(np.ravel(Y_True_data)) if x == 5]
    steady_idx = list(set(np.arange(0,len(Y_True_data))) - set(trans_idx))

    Y_pred = np.argmax(NN_model.predict(X_test), axis=-1)
    Y_test = np.ravel(Y_test)
    Y_pred = np.ravel(Y_pred)

    NN_overall_accuracy = accuracy_score(Y_test, Y_pred)
    NN_steady_accuracy = accuracy_score(Y_test[steady_idx], Y_pred[steady_idx])
    NN_trans_accuracy = accuracy_score(Y_test[trans_idx], Y_pred[trans_idx])

    text_file1 = base_path_dir + NN_saving_file + ".txt"
    msg1 = ' '.join([str(testing_subject),str(window_size),str(transition_point),str(phase_number),str(NN_steady_accuracy),str(NN_trans_accuracy),str(NN_overall_accuracy),"\n"])
    del [[X_train, Y_train, X_test, Y_test]]
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

    Y_steady_pred_result = []
    Y_steady_test_result = []
    Y_trans_pred_result = []
    Y_trans_test_result = []

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
                        Y = data.iloc[:, -1]

                        Y_True = data.iloc[:, -3]
                        trans_idx = [i for i, x in enumerate(Y_True) if x == 5]
                        steady_idx = list(set(np.arange(0,len(Y))) - set(trans_idx))

                        xg_test = xgb.DMatrix(X, label=Y)
                        Y_pred = model.predict(xg_test)

                        Y_pred_result = np.concatenate((Y_pred_result, Y_pred))
                        Y_test_result = np.concatenate((Y_test_result, Y))
                        Y_steady_pred_result = np.concatenate((Y_steady_pred_result, Y_pred[steady_idx]))
                        Y_steady_test_result = np.concatenate((Y_steady_test_result, Y[steady_idx])) 
                        Y_trans_pred_result = np.concatenate((Y_trans_pred_result, Y_pred[trans_idx])) 
                        Y_trans_test_result = np.concatenate((Y_trans_test_result, Y[trans_idx])) 

    for trial in trial_pool:
        train_path = fe_dir+"AB"+str(testing_subject)+"_LG_W"+str(window_size)+"_TP0_S2_R"+str(trial)+".csv"

        if path.exists(test_path) == 1:
            for test_read_path in glob.glob(test_path):
                data = pd.read_csv(test_read_path, header=None)
                X = data.iloc[:, :-3]
                Y = data.iloc[:, -1]
                xg_test = xgb.DMatrix(X, label=Y)
                Y_pred = model.predict(xg_test)

                Y_pred_result = np.concatenate((Y_pred_result, Y_pred))
                Y_test_result = np.concatenate((Y_test_result, Y))
                Y_steady_pred_result = np.concatenate((Y_steady_pred_result, Y_pred))
                Y_steady_test_result = np.concatenate((Y_steady_test_result, Y))
                del [[X, Y, Y_pred, xg_test]]

    Y_test_result = np.ravel(Y_test_result)
    Y_pred_result = np.ravel(Y_pred_result)
    Y_steady_test_result = np.ravel(Y_steady_test_result)
    Y_steady_pred_result = np.ravel(Y_steady_pred_result)
    Y_trans_test_result = np.ravel(Y_trans_test_result)
    Y_trans_pred_result = np.ravel(Y_trans_pred_result)

    XGB_overall_accuracy = accuracy_score(Y_test_result, Y_pred_result) 
    XGB_steady_accuracy = accuracy_score(Y_steady_test_result, Y_steady_pred_result)
    XGB_trans_accuracy = accuracy_score(Y_trans_test_result, Y_trans_pred_result)

    text_file1 = base_path_dir + XGB_saving_file + ".txt"
    msg1 = ' '.join([str(testing_subject),str(window_size),str(transition_point),str(phase_number),str(XGB_steady_accuracy),str(XGB_trans_accuracy),str(XGB_overall_accuracy),"\n"])
    return text_file1, msg1

run_combos = []
for testing_subject in [6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 27 ,28]:
    for window_size in [750]:
        for transition_point in [0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]:
            for phase_number in [1]:
                run_combos.append([testing_subject, window_size, transition_point, phase_number])
result = Parallel(n_jobs=-1)(delayed(lda_parallel)(combo) for combo in run_combos)
for r in result:
    with open(r[0],"a+") as f:
        f.write(r[1])

run_combos = []
for testing_subject in [6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 27 ,28]:
    for window_size in [350]:
        for transition_point in [0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]:
            for phase_number in [1]:
                for boost_round in [200]:
                    for tree_depth in [8]:
                        for child_weight in [0.01]:
                            run_combos.append([testing_subject, window_size, transition_point, phase_number, boost_round, tree_depth, child_weight])
result = Parallel(n_jobs=-1)(delayed(xgboost_parallel)(combo) for combo in run_combos)
for r in result:
    with open(r[0],"a+") as f:
        f.write(r[1])

run_combos = []
for testing_subject in [6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 27 ,28]:
    for window_size in [550]:
        for transition_point in [0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]:
            for phase_number in [1]:
                for layer_num in [4]:
                    for node_num in [25]:
                        for optimizer_value in ['SGD']:
                            run_combos.append([testing_subject, window_size, transition_point, phase_number, layer_num, node_num, optimizer_value])
result = Parallel(n_jobs=-1)(delayed(NN_parallel)(combo) for combo in run_combos)
for r in result:
    with open(r[0],"a+") as f:
        f.write(r[1])

run_combos = []
for testing_subject in [6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 27 ,28]:
    for window_size in [350]:
        for transition_point in [0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]:
            for phase_number in [1]:
                for kernel_type in ['rbf']:
                        run_combos.append([testing_subject, window_size, transition_point, phase_number, kernel_type])
result = Parallel(n_jobs=-1)(delayed(SVM_parallel)(combo) for combo in run_combos)
for r in result:
    with open(r[0],"a+") as f:
        f.write(r[1])

