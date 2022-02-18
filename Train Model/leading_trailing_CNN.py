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
from keras.layers import Dense, LSTM, Conv1D, Flatten, BatchNormalization, Dropout
from keras.optimizers import Adam, SGD, Nadam
from keras import backend as K
from keras import regularizers
from sklearn.model_selection import train_test_split, cross_val_score, KFold
import tensorflow as tf
from tensorflow.python.util import deprecation
from keras.initializers import he_uniform
from keras.layers import Activation
deprecation._PRINT_DEPRECATION_WARNINGS = False

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
    msg1 = ' '.join([str(testing_subject),str(window_size),str(transition_point),str(phase_number),str(start_leg),str(NN_steady_accuracy),str(NN_trans_accuracy),str(NN_overall_accuracy),"\n"])
    # del [[X_train, Y_train, X_test, Y_test]]
    # return text_file1, msg1
def build_cnn_model(window_size, conv_kernel, cnn_activation, dense_nodes, dense_layers, dense_optimizer):
    model = Sequential()
    model.add(Conv1D(14, conv_kernel,
                input_shape=(window_size, 14),
                kernel_initializer=he_uniform(seed=1),
                bias_initializer=he_uniform(seed=11)))
    output_kernel = window_size - conv_kernel + 1
    model.add(BatchNormalization(input_shape=((int)(output_kernel), 14)))
    model.add(Activation(cnn_activation))
    model.add(Dropout(0.2))

    model.add(Conv1D(14, conv_kernel,
                kernel_initializer=he_uniform(seed=44),
                bias_initializer=he_uniform(seed=32)))
    output_kernel = output_kernel - conv_kernel + 1
    model.add(BatchNormalization(input_shape=((int)(output_kernel), 14)))
    model.add(Dropout(0.2))

    model.add(Conv1D(14, conv_kernel,
                kernel_initializer=he_uniform(seed=21),
                bias_initializer=he_uniform(seed=56)))
    output_kernel = output_kernel - conv_kernel + 1
    model.add(BatchNormalization(input_shape=((int)(output_kernel), 14)))
    model.add(Dropout(0.2))
    model.add(Activation(cnn_activation))

    model.add(Conv1D(14, conv_kernel,
                kernel_initializer=he_uniform(seed=32),
                bias_initializer=he_uniform(seed=11)))
    output_kernel = output_kernel - conv_kernel + 1
    model.add(BatchNormalization(input_shape=((int)(output_kernel), 14)))
    model.add(Dropout(0.2))
    model.add(Activation(cnn_activation))

    model.add(Conv1D(14, conv_kernel,
                kernel_initializer=he_uniform(seed=32),
                bias_initializer=he_uniform(seed=11)))
    output_kernel = output_kernel - conv_kernel + 1
    model.add(BatchNormalization(input_shape=((int)(output_kernel), 14)))
    model.add(Dropout(0.2))
    model.add(Activation(cnn_activation))

    model.add(Flatten())

    while dense_layers:
        model.add(Dense(dense_nodes,
                    kernel_initializer=he_uniform(seed=74),
                    bias_initializer=he_uniform(seed=52)))
        dense_layers -= 1
    model.add(Dense(5, activation='softmax',
                kernel_initializer=he_uniform(seed=74),
                bias_initializer=he_uniform(seed=52)))
    model.compile(dense_optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

    # kernel_factor = 0.75
    # while layers:
    #     kernel_size = (int)(np.ceil(kernel_factor * conv_kernel))
    #     output_kernel -= kernel_size - 1
    #     model.add(Conv1D(10, kernel_size,
    #             input_shape=(window_size, 10),
    #             kernel_initializer=he_uniform(seed=1),
    #             bias_initializer=he_uniform(seed=11)))
    #     layers -= 1
    #     kernel_factor -= 0.25
def cnn_parallel(testing_subject, combo):
    # testing_subject = combo[0]
    window_size = combo[0]
    transition_point = combo[1]
    phase_number = combo[2]
    conv_kernel = combo[3]
    cnn_activation = combo[4]
    dense_layers = combo[5]
    dense_optimizer = combo[6]
    start_leg = combo[7]


    dense_nodes = (int)((window_size-5*conv_kernel + 5)/2)

    trial_pool = [1, 2, 3]
    subject_pool = [6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 27 ,28]
    del subject_pool[subject_pool.index(testing_subject)]

    X_train = np.empty((0,window_size,14))
    Y_train = np.empty((0,1))
    X_test = np.empty((0,window_size,14))
    Y_test = np.empty((0,1))
    Y_true = np.empty((0,1))

    for trial in trial_pool:
        for subject in subject_pool:
            for mode in training_mode:
                for starting_leg in start_leg:
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
        for starting_leg in start_leg:   
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
        ", Starting Leg: "+str(start_leg)+
        ", Steady Accuracy: "+str(CNN_steady_accuracy)+
        ", Trans Accuracy: "+str(CNN_trans_accuracy)+
        ", Overall Accuracy: "+str(CNN_overall_accuracy))

    text_file1 = base_path_dir + CNN_saving_file + ".txt"
    msg1 = ' '.join([str(testing_subject),str(window_size),str(transition_point),str(phase_number),str(conv_kernel),
        str(cnn_activation),str(dense_layers),str(dense_nodes),str(dense_optimizer),str(start_leg),
        str(CNN_steady_accuracy),str(CNN_trans_accuracy),str(CNN_overall_accuracy),"\n"])
    return text_file1, msg1

fe_dir = "/HDD/hipexo/Inseung/sim IMU feature extraction data_CNN/"
base_path_dir = "/HDD/hipexo/Inseung/Result/"


#######################################################################
# IMU Location Sweep
CNN_saving_file = "CNN_LeadTrail"
training_mode = ["RA2", "RA3", "RA4", "RA5", "RD2", "RD3", "RD4", "RD5", "SA1", "SA2", "SA3", "SA4", "SD1", "SD2", "SD3", "SD4"]
testing_mode = ["RA2", "RA3", "RA4", "RA5", "RD2", "RD3", "RD4", "RD5", "SA1", "SA2", "SA3", "SA4", "SD1", "SD2", "SD3", "SD4"]

for testing_subject in [6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 27 ,28]:
    run_combos = []
    for window_size in [100]:
        for transition_point in [0.2]:
            for phase_number in [1]:
                for conv_kernel in [10]:
                    for cnn_activation in ['relu']:
                        for dense_layers in [1]:
                            for dense_optimizer in ['adam']:
                                for start_leg in ['R', 'L']:
                                    run_combos.append([window_size, transition_point, phase_number, conv_kernel, cnn_activation, dense_layers, dense_optimizer, start_leg])
    result = Parallel(n_jobs=-1)(delayed(cnn_parallel)(testing_subject, combo) for combo in run_combos)
    for r in result:
        with open(r[0],"a+") as f:
            f.write(r[1])

#####################################################################
# Regular/Transition Point Sweeping
# CNN_saving_file = "CNN_transition_sweep"
# training_mode = ["RA2", "RA3", "RA4", "RA5", "RD2", "RD3", "RD4", "RD5", "SA1", "SA2", "SA3", "SA4", "SD1", "SD2", "SD3", "SD4"]
# testing_mode = ["RA2", "RA3", "RA4", "RA5", "RD2", "RD3", "RD4", "RD5", "SA1", "SA2", "SA3", "SA4", "SD1", "SD2", "SD3", "SD4"]

# for testing_subject in [6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 27 ,28]:
#     run_combos = []
#     for window_size in [100]:
#         for transition_point in [0.2]:
#             for phase_number in [1]:
#                 for conv_kernel in [10]:
#                     for cnn_activation in ['relu']:
#                         for dense_layers in [1]:
#                             for dense_optimizer in ['adam']:
#                                 run_combos.append([window_size, transition_point, phase_number, conv_kernel, cnn_activation, dense_layers, dense_optimizer])
#     result = Parallel(n_jobs=-1)(delayed(cnn_parallel)(testing_subject, combo) for combo in run_combos)
#     for r in result:
#         with open(r[0],"a+") as f:
#             f.write(r[1])
#####################################################################
# Leave Setting Analysis
# CNN_saving_file = "CNN_ramp_remove1"
# training_mode = ["RA3", "RA4", "RA5", "RD3", "RD4", "RD5", "SA1", "SA2", "SA3", "SA4", "SD1", "SD2", "SD3", "SD4"]
# testing_mode = ["RA2", "RD2"]

# run_combos = []
# for testing_subject in [6, 7, 8, 9, 10, 12, 13, 14, 15, 16]:
#     for window_size in [100]:
#         for transition_point in [0.2]:
#             for phase_number in [1]:
#                 for conv_kernel in [10]:
#                     for cnn_activation in ['relu']:
#                         for dense_layers in [1]:
#                             for dense_optimizer in ['adam']:
#                                 run_combos.append([testing_subject, window_size, transition_point, phase_number, conv_kernel, cnn_activation, dense_layers, dense_optimizer])
# result = Parallel(n_jobs=-1)(delayed(cnn_parallel)(combo) for combo in run_combos)
# for r in result:
#     with open(r[0],"a+") as f:
#         f.write(r[1])
# run_combos = []
# for testing_subject in [17, 18, 19, 20, 21, 23, 24, 25, 27 ,28]:
#     for window_size in [100]:
#         for transition_point in [0.2]:
#             for phase_number in [1]:
#                 for conv_kernel in [10]:
#                     for cnn_activation in ['relu']:
#                         for dense_layers in [1]:
#                             for dense_optimizer in ['adam']:
#                                 run_combos.append([testing_subject, window_size, transition_point, phase_number, conv_kernel, cnn_activation, dense_layers, dense_optimizer])
# result = Parallel(n_jobs=-1)(delayed(cnn_parallel)(combo) for combo in run_combos)
# for r in result:
#     with open(r[0],"a+") as f:
#         f.write(r[1])
