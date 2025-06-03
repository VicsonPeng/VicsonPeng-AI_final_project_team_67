#! /usr/bin/env python
#! coding:utf-8

import scipy.ndimage.interpolation as inter
from scipy.spatial.distance import cdist
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import pathlib
import copy
from scipy.signal import medfilt

def zoom(p, target_l=64, joints_num=25, joints_dim=2):
    """
    p: np.ndarray, shape = (原始帧数, joints_num, joints_dim)
       现在 joints_dim=2
    输出：插值或采样到 target_l 帧后的 np.ndarray，shape = (target_l, joints_num, 2)
    """
    p_copy = copy.deepcopy(p)
    l = p_copy.shape[0]
    p_new = np.empty([target_l, joints_num, joints_dim])
    for m in range(joints_num):
        for n in range(joints_dim):  # joints_dim=2 时循环 0,1
            p_copy[:, m, n] = medfilt(p_copy[:, m, n], 3)
            p_new[:, m, n] = inter.zoom(
                p_copy[:, m, n], target_l / l
            )[:target_l]
    return p_new

# 计算 JCD 特征（Joint‐to‐Joint Distance）
def norm_scale(x):
    return (x - np.mean(x)) / (np.mean(x) + 1e-8)  # 加 eps 避免 mean=0

def get_CG(p, C):
    """
    p: np.ndarray, shape = (frame_l, joint_n, joint_d=2)
    C.joint_n=21, C.frame_l=32, C.feat_d=210 (因为 C(21,2)=210)
    返回：M, shape = (frame_l, 210)
    """
    M_list = []
    # (i,j) 枚举上三角矩阵，不含对角线
    iu = np.triu_indices(C.joint_n, k=1, m=C.joint_n)
    for f in range(C.frame_l):
        # p[f] 形状 (21, 2)，cdist 会计算 2D 欧式距离
        dist_mat = cdist(p[f], p[f], 'euclidean')  # shape (21,21)
        d_m = dist_mat[iu]                         # 挑选 21×(21-1)/2 = 210 个值
        M_list.append(d_m)
    M = np.stack(M_list, axis=0)  # shape = (frame_l, 210)
    M = norm_scale(M)
    return M

def poses_diff(x):
    """
    x: Tensor, shape = (batch, frame_l, joint_n, joint_d=2)
    返回：同样 shape = (batch, frame_l, joint_n, joint_d=2)，
    代表“每帧 vs 下一帧”的差分后再上采样回原始长度。
    """
    _, H, W, _ = x.shape  # 假设 x.shape = (B, frame_l, joint_n, 2)
    # 先做相邻帧差分
    x = x[:, 1:, :, :] - x[:, :-1, :, :]   # shape = (B, frame_l-1, joint_n, 2)
    # 插值回 (frame_l, joint_n, 2)
    x = x.permute(0, 3, 1, 2)               # (B, 2, frame_l-1, joint_n)
    x = F.interpolate(
        x, size=(H, W), mode='bilinear', align_corners=False
    )                                       # (B, 2, frame_l, joint_n)
    x = x.permute(0, 2, 3, 1)               # (B, frame_l, joint_n, 2)
    return x

def poses_motion(P):
    """
    P: Tensor, shape = (batch, frame_l, joint_n, joint_d=2)
    slow: 每帧 vs 下一帧的 2D 差分后再插回 (batch, frame_l, joint_n*2)
    fast: 每隔 2 帧 vs 下一帧做差分 -> 然后插回 (batch, frame_l/2, joint_n*2)
    """
    # slow 差分
    P_diff_slow = poses_diff(P)                     # (B, frame_l, joint_n, 2)
    P_diff_slow = torch.flatten(P_diff_slow, start_dim=2)  
    # -> (B, frame_l, joint_n*2)

    # fast：先抽帧(隔 2)，再差分
    P_fast = P[:, ::2, :, :]                         # (B, frame_l//2, joint_n, 2)
    P_diff_fast = poses_diff(P_fast)                 # (B, frame_l//2, joint_n, 2)
    P_diff_fast = torch.flatten(P_diff_fast, start_dim=2)
    # -> (B, frame_l//2, joint_n*2)

    return P_diff_slow, P_diff_fast

def makedir(path):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
