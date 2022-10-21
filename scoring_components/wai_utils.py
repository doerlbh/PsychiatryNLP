from basic_utils import *

# global variables specific to the working alliance inventory

TASK_SCALE_POS = np.array([2,4,13,16,18,24,35]) - 1
TASK_SCALE_NEG = np.array([7,11,15,31,33]) - 1
TASK_SCALE = np.concatenate((TASK_SCALE_POS, TASK_SCALE_NEG))

BOND_SCALE_POS = np.array([5,8,17,19,21,23,26,28,36]) - 1
BOND_SCALE_NEG = np.array([1,20,29]) - 1
BOND_SCALE = np.concatenate((BOND_SCALE_POS, BOND_SCALE_NEG))

GOAL_SCALE_POS = np.array([6,14,22,25,30,32]) - 1
GOAL_SCALE_NEG = np.array([3,9,10,12,27,34]) - 1
GOAL_SCALE = np.concatenate((GOAL_SCALE_POS, GOAL_SCALE_NEG))

WAI_KEYS = np.ones(36)
WAI_KEYS[TASK_SCALE_NEG] = -1
WAI_KEYS[BOND_SCALE_NEG] = -1
WAI_KEYS[GOAL_SCALE_NEG] = -1

TASK_SCALE_KEYS = WAI_KEYS.copy()
mask = np.ones(36, dtype=bool)
mask[TASK_SCALE] = False
TASK_SCALE_KEYS[mask] = 0

BOND_SCALE_KEYS = WAI_KEYS.copy()
mask = np.ones(36, dtype=bool)
mask[BOND_SCALE] = False
BOND_SCALE_KEYS[mask] = 0

GOAL_SCALE_KEYS = WAI_KEYS.copy()
mask = np.ones(36, dtype=bool)
mask[GOAL_SCALE] = False
GOAL_SCALE_KEYS[mask] = 0

def get_wai_vectors(emb_type='doc2vec', score_type='standard'):

    if emb_type == 'doc2vec':
        if score_type == 'standard':
            wai_c_vectors = get_vectors('./doc2vec/vectors/wai_c_vectors.txt')
            wai_t_vectors = get_vectors('./doc2vec/vectors/wai_t_vectors.txt')
        elif score_type == 'counter':
            wai_c_vectors = [get_vectors('./doc2vec/vectors/wai_c_vectors.txt'), get_vectors('./doc2vec/vectors/wai_cn_vectors.txt')]
            wai_t_vectors = [get_vectors('./doc2vec/vectors/wai_t_vectors.txt'), get_vectors('./doc2vec/vectors/wai_tn_vectors.txt')]
    elif emb_type == 'sbert':
        if score_type == 'standard':
            wai_c_vectors = np.load('sbert/vectors/wai_c_vectors.npy')
            wai_t_vectors = np.load('sbert/vectors/wai_t_vectors.npy')
        elif score_type == 'counter':
            wai_c_vectors = [np.load('sbert/vectors/wai_c_vectors.npy'), np.load('sbert/vectors/wai_cn_vectors.npy')]
            wai_t_vectors = [np.load('sbert/vectors/wai_t_vectors.npy'), np.load('sbert/vectors/wai_tn_vectors.npy')]

    return wai_c_vectors, wai_t_vectors

def get_wai_scales(scores, scale='full'):
    if scale == 'full':
        scale_scores = np.matmul(scores, WAI_KEYS.reshape((-1, 1))).flatten()
    elif scale == 'task':
        scale_scores = np.matmul(scores, TASK_SCALE_KEYS.reshape((-1, 1))).flatten()
    elif scale == 'bond':
        scale_scores = np.matmul(scores, BOND_SCALE_KEYS.reshape((-1, 1))).flatten()
    elif scale == 'goal':
        scale_scores = np.matmul(scores, GOAL_SCALE_KEYS.reshape((-1, 1))).flatten()
    return scale_scores

def get_3scales(s1,s2,s3):
    news = []
    for i in range(len(s1)):
        news.append(np.array([s1[i], s2[i], s3[i]]).T)
    return news

