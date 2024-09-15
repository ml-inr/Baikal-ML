import numpy as np
from typing import Tuple

# Water properties
N = 1.37
COS_C = 1 / N
SIN_C = np.sqrt(1 - COS_C**2)
TAN_C = SIN_C / COS_C
C_PART = 299792458.
C_LIGHT = 218826621.

def get_smallest_tres(t_res: np.ndarray, n: int) -> np.ndarray:
    t_res = np.reshape( t_res, (-1,n), order='F' )
    t_res_idxs = np.argmin( np.abs( t_res ), axis=1, keepdims=True )
    t_res = np.take_along_axis( t_res, t_res_idxs, 1)
    return np.squeeze(t_res)

def eval_tres(
    mus: np.ndarray,
    om_coords: np.ndarray,
    t_det: np.ndarray,
    ev_starts: np.ndarray,
    mu_starts: np.ndarray
) -> np.ndarray:

    mus_dirs = np.array( (np.sin(mus[:,0])*np.cos(mus[:,1]), np.sin(mus[:,0])*np.sin(mus[:,1]), np.cos(mus[:,0])) )
    mus_dirs = np.transpose( mus_dirs, (1,0) )
    
    mus_point = mus[:, 2:5]
    targ_vect = om_coords - mus_point
    targ_dist = np.linalg.norm(targ_vect, axis=1, keepdims=True)
    dir_norm = np.linalg.norm(mus_dirs, axis=1, keepdims=True)
    
    cosAlpha = np.sum(targ_vect * mus_dirs / (1e-9 + targ_dist * dir_norm), axis=1, keepdims=True)
    cosAlpha2 = np.clip(cosAlpha**2, 0, 1)
    sinAlpha = np.sqrt(1 - cosAlpha2)
    
    dMuon = targ_dist * (cosAlpha - sinAlpha / TAN_C)
    tMuon = np.squeeze( 1e9 * dMuon / C_PART)
    dLight = targ_dist * sinAlpha / SIN_C
    tLight = np.squeeze( 1e9 * dLight / C_LIGHT )
    
    t_exp = tMuon + tLight + mus[:, 5]
    t_res_all = t_exp - t_det
    
    mu_ls = np.diff(mu_starts)
    ev_ls = np.diff(ev_starts)
    starts = np.concatenate( ([0],mu_ls*ev_ls), axis=0 ) 
    starts = np.cumsum(starts, axis=0)
    
    t_res = np.concatenate([ get_smallest_tres(t_res_all[starts[i]:starts[i+1]],mu_ls[i]) for i in range(len(ev_starts)-1) ], axis=0)
    
    return t_res

