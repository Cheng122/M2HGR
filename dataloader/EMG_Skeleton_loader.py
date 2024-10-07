from tqdm import tqdm
import numpy as np
from sklearn import preprocessing
import pickle
from pathlib import Path
import sys
sys.path.insert(0, '..')
from common.utils import *
current_file_dirpath = Path(__file__).parent.parent.absolute()


def load_EMG_Skeleton_data(
        train_path=current_file_dirpath / Path("dataset/train.pkl"),
        test_path=current_file_dirpath / Path("dataset/test.pkl")):

    Train = pickle.load(open(train_path, "rb"))
    Test = pickle.load(open(test_path, "rb"))

    # create LabelEncoder object
    le = preprocessing.LabelEncoder()

    le.fit(Train['label']) 

    print("Loading EMG_skeleton Dataset")
    return Train, Test, le


class ESConfig():
    def __init__(self):
        self.frame_l = 32                       # the length of frames
        self.joint_n = 21                       # the number of joints
        self.joint_d = 2                        # the dimension of joints
        self.clc_num = 15                       # the number of class
        self.feat_d = 210
        self.filters = 64
        self.emgdata = 8                        # the number of EMG channels, example: [291.0, -20.0, 185.0, 6.0, 26.0, 26.0, -453.0, 506.0]

# Genrate dataset
# T: Dataset  C:config   le:labelEncoder


def ESdata_generator(T, C, le):
    X_0 = []                                    # JCD data
    X_1 = []                                    # skeleton data: (target_frame, joint_num, joint_coords_dims)
    X_2 = []                                    # emg data: (target_frame, emgdata)
    Y = []                                      

    labels = le.transform(T['label'])

    for i in tqdm(range(len(T['pose']))):
        p = np.copy(T['pose'][i])               # p.shape = (frame,joint_num,joint_coords_dims)
        p = zoom(p, target_l=C.frame_l,
                 joints_num=C.joint_n, joints_dim=C.joint_d)    # p.shape (target_frame,joint_num,joint_coords_dims)
        e = np.copy(T['EMG'][i])
        e = zoom_EMG(e, target_l=C.frame_l, EMG_dim=C.emgdata)
        
        label = labels[i]
        M = get_CG(p, C)                        # M.shape = (target_frame,(joint_num - 1) * joint_num / 2)

        X_0.append(M)
        X_1.append(p)
        X_2.append(e)
        Y.append(label)

    X_0 = np.stack(X_0)
    X_1 = np.stack(X_1)
    X_2 = np.stack(X_2)
    Y = np.stack(Y)
    return X_0, X_1, X_2, Y
