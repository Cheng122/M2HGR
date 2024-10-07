import scipy.ndimage.interpolation as inter
from scipy.spatial.distance import cdist
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import pathlib
import copy
from scipy.signal import medfilt

# Temple resizing function
# interpolate l frames to target_l frames

def zoom(p,target_l=64,joints_num=21,joints_dim=2):
    l = np.array(p).shape[0]
    p = np.array(p)
    p_new = np.empty([target_l,joints_num,joints_dim]) 
    for m in range(joints_num):
        for n in range(joints_dim):
            p[:,m,n] = medfilt(p[:,m,n],3)
            p_new[:,m,n] = inter.zoom(p[:,m,n],target_l/l)[:target_l]         
    return p_new


# Get the target EMGdata
def zoom_EMG(N, target_l, EMG_dim=8):
    N_copy = copy.deepcopy(N)
    l = N_copy.shape[0]
    N_new = np.empty([target_l, EMG_dim])
    for m in range(EMG_dim):
        N_copy[:, m] = medfilt(N_copy[:, m], 3)
        N_new[:, m] = inter.zoom(N_copy[:, m], target_l/l)[:target_l]
    return N_new


# Calculate JCD feature: normalization
def norm_scale(x):
    return (x-np.mean(x))/np.mean(x)


# Calculate JCD feature
def get_CG(p, C):
    M = []
    # upper triangle index with offset 1, which means upper triangle without diagonal
    iu = np.triu_indices(C.joint_n, 1, C.joint_n)                                   # return the indices of upper triangle matrix
    for f in range(C.frame_l):
        d_m = cdist(p[f], p[f], 'euclidean')                                        # return the Euclidean Distance of p[f] & p[f]，shape=(C.joint_n,C.joint_n)
        d_m = d_m[iu]                                                               # get the flatten vector of the upper triangle matrix, shape(C.joint_n*C.joint_n,)
        M.append(d_m)
    M = np.stack(M)
    M = norm_scale(M)
    return M


def poses_diff(x):
    _, H, W, _ = x.shape

    # x.shape (batch,channel,joint_num,joint_dim)
    x = x[:, 1:, ...] - x[:, :-1, ...]

    # x.shape (batch,joint_dim,channel,joint_num,)
    x = x.permute(0, 3, 1, 2)
    x = F.interpolate(x, size=(H, W),
                      align_corners=False, mode='bilinear')
    x = x.permute(0, 2, 3, 1)
    # x.shape (batch,channel,joint_num,joint_dim)
    return x


def poses_motion(P):
    # different from the original version
    # TODO: check the funtion, make sure it's right
    P_diff_slow = poses_diff(P)
    P_diff_slow = torch.flatten(P_diff_slow, start_dim=2)
    P_fast = P[:, ::2, :, :]
    P_diff_fast = poses_diff(P_fast)
    P_diff_fast = torch.flatten(P_diff_fast, start_dim=2)
    # return (B,target_l,joint_d * joint_n) , (B,target_l/2,joint_d * joint_n)
    return P_diff_slow, P_diff_fast


def EMG_diff(N):
    batchsize, frames, channels = N.shape
    N = N.unsqueeze(-1)                                 # batchsize, frames, channles, 1
    N_diff = N[:, 1:, ...] - N[:, :-1, ...]
    N_diff = N_diff.permute(0, 3, 1, 2)                 # bs, 1, f, c
    N_diff = F.interpolate(N_diff, size=(frames, channels),
                      align_corners=False, mode='bilinear')    
    N_diff = N_diff.permute(0, 2, 3, 1)                 # bs, f, c, 1      
    N_diff = N_diff.squeeze(-1)         
    return N_diff


def makedir(path):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
