from matplotlib import pyplot as plt
from scipy import spatial
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from dtw import dtw
import umap
from sklearn.manifold import MDS, TSNE, Isomap, SpectralEmbedding
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, k_means
from statannotations.Annotator import Annotator
from scipy.interpolate import interp1d, make_interp_spline

from wai_utils import *

def get_score(vector, ref_vectors, score_type='standard'):
    if score_type == 'standard':
        score = [1 - spatial.distance.cosine(vector, v) for v in ref_vectors]
    elif score_type == 'counter':
        sim_p = [1 - spatial.distance.cosine(vector, v) for v in ref_vectors[0]]
        sim_n = [1 - spatial.distance.cosine(vector, v) for v in ref_vectors[1]]
        score = [a - b for a, b in zip(sim_p, sim_n)]
    return score

def get_scores(vectors, ref_vectors, score_type='standard'):
    scores = np.array([get_score(v, ref_vectors, score_type) for v in vectors])
    return scores

def get_scales(scores, scale='full', inventory='wai', score_type='standard'):
    if inventory=='wai':
        scale_scores = get_wai_scales(scores, scale)
    return scale_scores

def get_sessions(measures, indices):
    sessions = [list(measures[i[0]:i[1]]) for i in indices]
    return sessions
  
def numpy_fillna(data):
    # Get lengths of each row of data
    lens = np.array([len(i) for i in data])

    # Mask of valid places in each row
    mask = np.arange(lens.max()) < lens[:,None]

    # Setup output array and put elements from data into masked positions
    out = np.empty(mask.shape, dtype=data.dtype)
    out[:] = np.nan
    out[mask] = np.concatenate(data)
    return out

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def get_dtw(sessions, upbound=-1):
    sessions = np.array(sessions)
    D = - np.ones((len(sessions), len(sessions)))
    print(D.shape)
    for i, s1 in enumerate(sessions):
        template = np.array(s1)
        for j, s2 in enumerate(sessions):
            print("finished "+str(i)+" "+str(j), end="\r")
            if D[i,j] == -1:
                query = np.array(s2)
                d = dtw(query[:upbound], template[:upbound], keep_internals=True).distance
                D[i,j] = d
                D[j,i] = d
    return D

def get_all_scales_and_sessions(emb_type, turn_owner, vectors, ref_vectors, indices, score_type='standard', mode='compute', prefix=''):

    if mode == 'load':
        scores = np.load(f'{emb_type}/scores/{prefix}scores_{turn_owner}.npy', allow_pickle=True)
    else:
        scores = get_scores(vectors, ref_vectors, score_type)   
    full_scale = get_scales(scores, scale='full')
    task_scale = get_scales(scores, scale='task')
    bond_scale = get_scales(scores, scale='bond')
    goal_scale = get_scales(scores, scale='goal')
    scales = [full_scale, task_scale, bond_scale, goal_scale]

    if mode == 'load':
        sessions_all = np.load(f'{emb_type}/sessions/{prefix}sessions_all_{turn_owner}.npy', allow_pickle=True)
        sessions_full_scale = np.load(f'{emb_type}/sessions/{prefix}sessions_full_scale_{turn_owner}.npy', allow_pickle=True)
        sessions_task_scale = np.load(f'{emb_type}/sessions/{prefix}sessions_task_scale_{turn_owner}.npy', allow_pickle=True)
        sessions_bond_scale = np.load(f'{emb_type}/sessions/{prefix}sessions_bond_scale_{turn_owner}.npy', allow_pickle=True)
        sessions_goal_scale = np.load(f'{emb_type}/sessions/{prefix}sessions_goal_scale_{turn_owner}.npy', allow_pickle=True)
    else:
        sessions_all = get_sessions(scores, indices)
        sessions_full_scale = get_sessions(full_scale, indices)
        sessions_task_scale = get_sessions(task_scale, indices)
        sessions_bond_scale = get_sessions(bond_scale, indices)
        sessions_goal_scale = get_sessions(goal_scale, indices)
    
        if mode == 'save':
            np.save(f'{emb_type}/scores/{prefix}scores_{turn_owner}.npy', scores)
            np.save(f'{emb_type}/sessions/{prefix}sessions_all_{turn_owner}.npy', sessions_all)
            np.save(f'{emb_type}/sessions/{prefix}sessions_full_scale_{turn_owner}.npy', sessions_full_scale)
            np.save(f'{emb_type}/sessions/{prefix}sessions_task_scale_{turn_owner}.npy', sessions_task_scale)
            np.save(f'{emb_type}/sessions/{prefix}sessions_bond_scale_{turn_owner}.npy', sessions_bond_scale)
            np.save(f'{emb_type}/sessions/{prefix}sessions_goal_scale_{turn_owner}.npy', sessions_goal_scale)
            
    sessions = [sessions_all, sessions_full_scale, sessions_task_scale, sessions_bond_scale, sessions_goal_scale]

    return scales, sessions, scores

def get_df_from_dicts(dict_t, dict_c):
    data = {}
    for k in dict_t.keys():
        data[k] = list(dict_t[k]) + list(dict_c[k])
    df = pd.DataFrame(data)
    return df
