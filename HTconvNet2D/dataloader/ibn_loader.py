#! /usr/bin/env python
#! coding:utf-8:w

import numpy as np
from sklearn import preprocessing
import pickle
from pathlib import Path
from tqdm import tqdm
import sys
sys.path.insert(0, '..')
from utils import *  # noqa
current_file_dirpath = Path(__file__).parent.absolute()


def load_ibn_data(
        train_path=current_file_dirpath / Path("C:/Users/vicso/Source/Repos/AI_final_project_hand_gesture_recognization/HT_convNET_git/ipn_pkl_2d/train.pkl"),
        
        test_path=current_file_dirpath / Path("C:/Users/vicso/Source/Repos/AI_final_project_hand_gesture_recognization/HT_convNET_git/ipn_pkl_2d/test.pkl"),
):
    Train = pickle.load(open(train_path, "rb"))
    Test = pickle.load(open(test_path, "rb"))
    return Train, Test, None


class IConfig():
    def __init__(self):
        self.frame_l = 32  # the length of frames
        self.joint_n = 21  # the number of joints
        self.joint_d = 2  # the dimension of joints
        self.class_num = 14
        self.feat_d = 210
        self.filters = 64


class Idata_generator:
    def __init__(self, label_level='label'):
        self.label_level = label_level

    def __call__(self, T, C, le=None):
        X_0 = []
        X_1 = []
        Y = []
        for i in tqdm(range(len(T['pose']))):
            p = np.copy(T['pose'][i].reshape([-1, C.joint_n, C.joint_d]))

            # 跳過全為 0 的 sample
            if np.all(p == 0):
                continue

            p = zoom(p, target_l=C.frame_l,
                    joints_num=C.joint_n, joints_dim=C.joint_d)
            label = (T[self.label_level])[i] - 1
            M = get_CG(p, C)
            X_0.append(M)
            X_1.append(p)
            Y.append(label)
        self.X_0 = np.stack(X_0)
        self.X_1 = np.stack(X_1)
        self.Y = np.stack(Y)
        return self.X_0, self.X_1, self.Y



if __name__ == '__main__':
    Train, _ = load_ibn_data()
    C = IConfig()
    X_0, X_1, Y = Idata_generator('label')(Train, C, 'label')
    print(Y)
    print("X_0.shape", X_0.shape)
    print("X_1.shape", X_1.shape)
