from train_LDA import *
from train_SVM import *
from train_NN import *
from train_xgboost import *

LDA_saving_file = "LDA_remove2"
SVM_saving_file = "SVM_remove2"
NN_saving_file = "NN_remove2"
XGB_saving_file = "XGB_remove2"

training_mode = ["RA2", "RA4", "RA5", "RD2", "RD4", "RD5","SA1", "SA3", "SA4", "SD1", "SD3", "SD4"]
testing_mode = ["RA3", "RD3", "SA2", "SD2"]
# training_mode = ["RA2", "RA3", "RA4", "RA5", "RD2", "RD3", "RD4", "RD5","SA1", "SA2", "SA3", "SA4", "SD1", "SD2", "SD3", "SD4"]:
# testing_mode = ["RA2", "RA3", "RA4", "RA5", "RD2", "RD3", "RD4", "RD5","SA1", "SA2", "SA3", "SA4", "SD1", "SD2", "SD3", "SD4"]:

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

# run_combos = []
# for testing_subject in [6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 27 ,28]:
#     for window_size in [350]:
#         for transition_point in [0.2]:
#             for phase_number in [1]:
#                 for kernel_type in ['rbf']:
#                         run_combos.append([testing_subject, window_size, transition_point, phase_number, kernel_type])

# result = Parallel(n_jobs=-1)(delayed(SVM_parallel)(combo) for combo in run_combos)
# for r in result:
#     with open(r[0],"a+") as f:
#         f.write(r[1])

# run_combos = []
# for testing_subject in [6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 27 ,28]:
#     for window_size in [550]:
#         for transition_point in [0.2]:
#             for phase_number in [1]:
#                 for layer_num in [4]:
#                     for node_num in [25]:
#                         for optimizer_value in ['SGD']:
#                             run_combos.append([testing_subject, window_size, transition_point, phase_number, layer_num, node_num, optimizer_value])

# result = Parallel(n_jobs=-1)(delayed(NN_parallel)(combo) for combo in run_combos)
# for r in result:
#     with open(r[0],"a+") as f:
#         f.write(r[1])

# run_combos = []
# for testing_subject in [6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 27 ,28]:
#     for window_size in [350]:
#         for transition_point in [0.2]:
#             for phase_number in [1]:
#                 for boost_round in [200]:
#                     for tree_depth in [8]:
#                         for child_weight in [0.01]:
#                             run_combos.append([testing_subject, window_size, transition_point, phase_number, boost_round, tree_depth, child_weight])

# result = Parallel(n_jobs=-1)(delayed(xgboost_parallel)(combo) for combo in run_combos)
# for r in result:
#     with open(r[0],"a+") as f:
#         f.write(r[1])