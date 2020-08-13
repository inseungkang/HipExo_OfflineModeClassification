import os
from os import path

for subject in [6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 23, 24]:
    base_path = "/HDD/Inseung/Dropbox (GaTech)/ML/data/sensor_fusion/New Feature Data_Processed/"
    # base_path = "F:\\Dropbox (GaTech)\\ML\\data\\sensor_fusion\\opensim feature test\\"
    os.makedirs(base_path+"AB"+str(subject))