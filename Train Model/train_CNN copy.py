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
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore',category=FutureWarning)

fe_dir = "/Users/inseungkang/Documents/hipexo_ml/feature extraction data_CNN/"
base_path_dir = "/Users/inseungkang/Documents/hipexo_ml/Result/"

subject = 6
mode = "RA2"
transition_point = 0.2
starting_leg = "R"
trial = 1

data_path = fe_dir+"AB"+str(subject)+"_"+str(mode)+"_TP"+str(int(transition_point*100))+"_S2_"+str(starting_leg)+str(trial)+"_CNN.csv"

if path.exists(data_path) == 1:
    for train_read_path in glob.glob(data_path):
        data = pd.read_csv(train_read_path, header=None)
        Z = data.iloc[:, :-3].to_numpy()
        Y = data.iloc[:, -1].to_numpy()

from numpy import sum,isrealobj,sqrt
from numpy.random import standard_normal
def awgn(s,SNRdB,L=1):
    gamma = 10**(SNRdB/10) #SNR to linear scale
    if s.ndim==1:# if s is single dimensional vector
        P=L*sum(abs(s)**2)/len(s) #Actual power in the vector
    else: # multi-dimensional signals like MFSK
        P=L*sum(sum(abs(s)**2))/len(s) # if s is a matrix [MxN]
    N0=P/gamma # Find the noise spectral density
    if isrealobj(s):# check if input is real/complex object type
        n = sqrt(N0/2)*standard_normal(s.shape) # computed noise
    else:
        n = sqrt(N0/2)*(standard_normal(s.shape)+1j*standard_normal(s.shape))
    r = s + n # received signal
    return r

out = np.empty(Z.shape)
for ii in np.arange(14):
    out[:,ii] = awgn(Z[:,ii],20)

print(Z[0])
plt.plot(X[:,0])
plt.plot(Z[:,0])
plt.show()

def CNN_Train():
    trial_pool = [1, 2, 3]
    subject_pool = [6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 27 ,28]
    del subject_pool[subject_pool.index(testing_subject)]

    X_train = np.empty((0,window_size,14))
    Y_train = np.empty((0,1))
    X_test = np.empty((0,window_size,14))
    Y_test = np.empty((0,1))
    Y_True_data = pd.DataFrame()

    Y_steady_pred_result = []
    Y_steady_test_result = []
    Y_trans_pred_result = []
    Y_trans_test_result = []

    for trial in trial_pool:
        for subject in subject_pool:
            for mode in training_mode:
                for starting_leg in ["R", "L"]:
                    train_path = fe_dir+"AB"+str(subject)+"_"+str(mode)+"_TP"+str(int(transition_point*100))+"_S2_"+str(starting_leg)+str(trial)+"_CNN.csv"

                    if path.exists(train_path) == 1:
                        for train_read_path in glob.glob(train_path):
                            data = pd.read_csv(train_read_path, header=None)
                            X = data.iloc[:, :-3].to_numpy()
                            Y = data.iloc[:, -1].to_numpy()

                            shape = (X.shape[0] - window_size + 1, window_size, X.shape[1])
                            strides = (X.strides[0], X.strides[0], X.strides[1])
                            X = np.lib.stride_tricks.as_strided(X, shape=shape, strides=strides)[::10]
                            Y = np.expand_dims(Y[window_size - 1:][::10], axis=1)

                            X_train = np.concatenate([X_train, X], axis=0)
                            Y_train = np.concatenate([Y_train, Y], axis=0)

            train_path = fe_dir+"AB"+str(subject)+"_LG_TP0_S2_R"+str(trial)+"_CNN.csv"
            if path.exists(train_path) == 1:
                for train_read_path in glob.glob(train_path):
                    data = pd.read_csv(train_read_path, header=None)
                    X = data.iloc[:, :-3].to_numpy()
                    Y = data.iloc[:, -1].to_numpy()

                    shape = (X.shape[0] - window_size + 1, window_size, X.shape[1])
                    strides = (X.strides[0], X.strides[0], X.strides[1])
                    X = np.lib.stride_tricks.as_strided(X, shape=shape, strides=strides)[::10]
                    Y = np.expand_dims(Y[window_size - 1:][::10], axis=1)

                    X_train = np.concatenate([X_train, X], axis=0)
                    Y_train = np.concatenate([Y_train, Y], axis=0)


    for mode in testing_mode:
        for starting_leg in ["R", "L"]:   
            for trial in trial_pool: 
                test_path = fe_dir+"AB"+str(subject)+"_"+str(mode)+"_TP"+str(int(transition_point*100))+"_S2_"+str(starting_leg)+str(trial)+"_CNN.csv"

                if path.exists(test_path) == 1:
                    for test_read_path in glob.glob(test_path):
                        data = pd.read_csv(test_read_path, header=None)
                        X = data.iloc[:, :-3].to_numpy()
                        Y = data.iloc[:, -1].to_numpy()
                        Y_True = data.iloc[:, -3]
                        Y_True_data = pd.concat([Y_True_data, Y_True], axis=0, ignore_index=True)

                        shape = (X.shape[0] - window_size + 1, window_size, X.shape[1])
                        strides = (X.strides[0], X.strides[0], X.strides[1])
                        X = np.lib.stride_tricks.as_strided(X, shape=shape, strides=strides)[::10]
                        Y = np.expand_dims(Y[window_size - 1:][::10], axis=1)
                        X_test = np.concatenate([X_test, X], axis=0)
                        Y_test = np.concatenate([Y_test, Y], axis=0)

    for trial in trial_pool:
        train_path = fe_dir+"AB"+str(subject)+"_LG_TP0_S2_R"+str(trial)+"_CNN.csv"

        if path.exists(test_path) == 1:
            for test_read_path in glob.glob(test_path):
                data = pd.read_csv(test_read_path, header=None)
                X = data.iloc[:, :-3].to_numpy()
                Y = data.iloc[:, -1].to_numpy()
                Y_True = data.iloc[:, -3]
                Y_True_data = pd.concat([Y_True_data, Y_True], axis=0, ignore_index=True)

                shape = (X.shape[0] - window_size + 1, window_size, X.shape[1])
                strides = (X.strides[0], X.strides[0], X.strides[1])
                X = np.lib.stride_tricks.as_strided(X, shape=shape, strides=strides)[::10]
                Y = np.expand_dims(Y[window_size - 1:][::10], axis=1)
                X_test = np.concatenate([X_test, X], axis=0)
                Y_test = np.concatenate([Y_test, Y], axis=0)

    del [[X, Y]]
    CNN_model = build_cnn_model(window_size, 14, conv_kernel, cnn_activation, dense_layers, dense_optimizer)
    CNN_model.fit(X_train, Y_train, epochs=200, batch_size=128, verbose=0, validation_data=(X_test, Y_test),shuffle=True,callbacks=[EarlyStopping(patience=100,restore_best_weights=True)])

    trans_idx = [i for i, x in enumerate(np.ravel(Y_True_data)) if x == 5]
    steady_idx = list(set(np.arange(0,len(Y_True_data))) - set(trans_idx))

    Y_pred = np.argmax(CNN_model.predict(X_test), axis=-1)
    Y_test = np.ravel(Y_test)
    Y_pred = np.ravel(Y_pred)

    NN_overall_accuracy = accuracy_score(Y_test, Y_pred)
    # NN_steady_accuracy = accuracy_score(Y_test[steady_idx], Y_pred[steady_idx])
    # NN_trans_accuracy = accuracy_score(Y_test[trans_idx], Y_pred[trans_idx])
    print(NN_overall_accuracy)
    text_file1 = base_path_dir + CNN_saving_file + ".txt"
    msg1 = ' '.join([str(testing_subject),str(window_size),str(transition_point),str(phase_number),str(NN_steady_accuracy),str(NN_trans_accuracy),str(NN_overall_accuracy),"\n"])

def cnn_parallel(combo):
    testing_subject = combo[0]
    window_size = combo[1]
    transition_point = combo[2]
    phase_number = combo[3]
    conv_kernel = combo[4]
    cnn_activation = combo[5]
    dense_layers = combo[6]
    dense_optimizer = combo[7]

    dense_nodes = (int)((window_size-5*conv_kernel + 5)/2)

    trial_pool = [1, 2, 3]
    subject_pool = [6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 27 ,28]
    del subject_pool[subject_pool.index(testing_subject)]

    X_train = np.empty((0,window_size,14))
    Y_train = np.empty((0,1))
    X_test = np.empty((0,window_size,14))
    Y_test = np.empty((0,1))
    Y_true = np.empty((0,1))

    # Y_steady_pred_result = []
    # Y_steady_test_result = []
    # Y_trans_pred_result = []
    # Y_trans_test_result = []

    for trial in trial_pool:
        for subject in subject_pool:
            for mode in training_mode:
                for starting_leg in ["R", "L"]:
                    train_path = fe_dir+"AB"+str(subject)+"_"+str(mode)+"_TP"+str(int(transition_point*100))+"_S2_"+str(starting_leg)+str(trial)+"_CNN.csv"

                    if path.exists(train_path) == 1:
                        for train_read_path in glob.glob(train_path):
                            data = pd.read_csv(train_read_path, header=None)
                            X = data.iloc[:, :-3].to_numpy()
                            Y = data.iloc[:, -1].to_numpy()

                            shape = (X.shape[0] - window_size + 1, window_size, X.shape[1])
                            strides = (X.strides[0], X.strides[0], X.strides[1])
                            X = np.lib.stride_tricks.as_strided(X, shape=shape, strides=strides)[::10]
                            Y = np.expand_dims(Y[window_size - 1:][::10], axis=1)

                            X_train = np.concatenate([X_train, X], axis=0)
                            Y_train = np.concatenate([Y_train, Y], axis=0)

            train_path = fe_dir+"AB"+str(subject)+"_LG_TP0_S2_R"+str(trial)+"_CNN.csv"
            if path.exists(train_path) == 1:
                for train_read_path in glob.glob(train_path):
                    data = pd.read_csv(train_read_path, header=None)
                    X = data.iloc[:, :-3].to_numpy()
                    Y = data.iloc[:, -1].to_numpy()

                    shape = (X.shape[0] - window_size + 1, window_size, X.shape[1])
                    strides = (X.strides[0], X.strides[0], X.strides[1])
                    X = np.lib.stride_tricks.as_strided(X, shape=shape, strides=strides)[::10]
                    Y = np.expand_dims(Y[window_size - 1:][::10], axis=1)

                    X_train = np.concatenate([X_train, X], axis=0)
                    Y_train = np.concatenate([Y_train, Y], axis=0)


    for mode in testing_mode:
        for starting_leg in ["R", "L"]:   
            for trial in trial_pool: 
                test_path = fe_dir+"AB"+str(subject)+"_"+str(mode)+"_TP"+str(int(transition_point*100))+"_S2_"+str(starting_leg)+str(trial)+"_CNN.csv"

                if path.exists(test_path) == 1:
                    for test_read_path in glob.glob(test_path):
                        data = pd.read_csv(test_read_path, header=None)
                        X = data.iloc[:, :-3].to_numpy()
                        Y = data.iloc[:, -1].to_numpy()
                        Y_t = data.iloc[:, -3].to_numpy()

                        shape = (X.shape[0] - window_size + 1, window_size, X.shape[1])
                        strides = (X.strides[0], X.strides[0], X.strides[1])
                        X = np.lib.stride_tricks.as_strided(X, shape=shape, strides=strides)[::10]
                        Y = np.expand_dims(Y[window_size - 1:][::10], axis=1)
                        Y_t = np.expand_dims(Y_t[window_size - 1:][::10], axis=1)
                        X_test = np.concatenate([X_test, X], axis=0)
                        Y_test = np.concatenate([Y_test, Y], axis=0)
                        Y_true = np.concatenate([Y_true, Y_t], axis=0)

    for trial in trial_pool:
        train_path = fe_dir+"AB"+str(subject)+"_LG_TP0_S2_R"+str(trial)+"_CNN.csv"

        if path.exists(test_path) == 1:
            for test_read_path in glob.glob(test_path):
                data = pd.read_csv(test_read_path, header=None)
                X = data.iloc[:, :-3].to_numpy()
                Y = data.iloc[:, -1].to_numpy()
                Y_t = data.iloc[:, -3].to_numpy()

                shape = (X.shape[0] - window_size + 1, window_size, X.shape[1])
                strides = (X.strides[0], X.strides[0], X.strides[1])
                X = np.lib.stride_tricks.as_strided(X, shape=shape, strides=strides)[::10]
                Y = np.expand_dims(Y[window_size - 1:][::10], axis=1)
                Y_t = np.expand_dims(Y_t[window_size - 1:][::10], axis=1)
                X_test = np.concatenate([X_test, X], axis=0)
                Y_test = np.concatenate([Y_test, Y], axis=0)
                Y_true = np.concatenate([Y_true, Y_t], axis=0)

    del [[X, Y]]
    CNN_model = build_cnn_model(window_size, conv_kernel, cnn_activation, dense_nodes, dense_layers, dense_optimizer)
    CNN_model.fit(X_train, Y_train, epochs=200, batch_size=128, verbose=0, validation_split=0.2, shuffle=True, callbacks=[EarlyStopping(patience=100,restore_best_weights=True)])

    trans_idx = [i for i, x in enumerate(np.ravel(Y_true)) if x == 5]
    steady_idx = list(set(np.arange(0,len(Y_true))) - set(trans_idx))

    Y_pred = np.argmax(CNN_model.predict(X_test), axis=-1)
    Y_test = np.ravel(Y_test)
    Y_pred = np.ravel(Y_pred)

    CNN_steady_accuracy = accuracy_score(Y_test[steady_idx], Y_pred[steady_idx])
    CNN_trans_accuracy = accuracy_score(Y_test[trans_idx], Y_pred[trans_idx])
    CNN_overall_accuracy = accuracy_score(Y_test, Y_pred)

    del [[X_train, X_test, Y_train, Y_test]]    
    print("Testing Subject: "+str(testing_subject)+
        ", Window Size: "+str(window_size)+
        ", Kernel Size: "+str(conv_kernel)+
        ", CNN Activation: "+str(cnn_activation)+
        ", Dense Layer: "+str(dense_layers)+
        ", Dense Node: "+str(dense_nodes)+
        ", Dense Optimizer: "+str(dense_optimizer)+
        ", Steady Accuracy: "+str(CNN_steady_accuracy)+
        ", Trans Accuracy: "+str(CNN_trans_accuracy)+
        ", Overall Accuracy: "+str(CNN_overall_accuracy))

    text_file1 = base_path_dir + CNN_saving_file + ".txt"
    msg1 = ' '.join([str(testing_subject),str(window_size),str(transition_point),str(phase_number),str(conv_kernel),
        str(cnn_activation),str(dense_layers),str(dense_nodes),str(dense_optimizer),str(CNN_steady_accuracy),str(CNN_trans_accuracy),str(CNN_overall_accuracy),"\n"])
    return text_file1, msg1



#####################################################################
CNN_saving_file = "CNN_transition_sweep"
training_mode = ["RA2", "RA3", "RA4", "RA5", "RD2", "RD3", "RD4", "RD5", "SA1", "SA2", "SA3", "SA4", "SD1", "SD2", "SD3", "SD4"]
testing_mode = ["RA2", "RA3", "RA4", "RA5", "RD2", "RD3", "RD4", "RD5", "SA1", "SA2", "SA3", "SA4", "SD1", "SD2", "SD3", "SD4"]
#####################################################################

