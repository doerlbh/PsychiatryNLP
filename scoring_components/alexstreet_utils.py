from inference_utils import *

filepath = './data/alexstreet/therapist/lines.list'
num_lines = []
with open(filepath) as fp:
    for line in fp:
        num_lines.append(int(line))
indices = np.concatenate(((0,), np.cumsum(num_lines))) 
index_pairs = [(indices[i], indices[i+1]) for i in np.arange(len(num_lines))]

# print(np.where(np.array(num_lines) == 0))
# print(np.where(np.array(num_lines) == 1))

filepath = './data/alexstreet/patient/lines.list'
num_lines = []
with open(filepath) as fp:
    for line in fp:
        num_lines.append(int(line))
indices = np.concatenate(((0,), np.cumsum(num_lines))) 
index_pairs = [(indices[i], indices[i+1]) for i in np.arange(len(num_lines))]

# print(np.where(np.array(num_lines) == 0))
# print(np.where(np.array(num_lines) == 1))

# Get rid of sessions 115 (ANXI), 163 (ANXI), 281 (ANXI), 569 (DEPR), 619 (DEPR) 
# and 524 (DEPR), 804 (DEPR) since they are empty.

ANXI_INDICES = np.arange(498) - 3
DEPR_INDICES = np.arange(377) + 498 - 3 - 4
SCHI_INDICES = np.arange(71) + 498 + 377 - 3 - 4
SUIC_INDICES = np.arange(12) + 498 + 377 + 71 - 3 - 4

DISORDER_INDICES = ['anxiety'] * (498 - 3) + ['depression'] * (377 - 4) + ['schizophrenia'] * 71 + ['suicidal'] * 12

ANXI_INDICES_FULL = np.arange(498)
DEPR_INDICES_FULL = np.arange(377) + 498 
SCHI_INDICES_FULL = np.arange(71) + 498 + 377 
SUIC_INDICES_FULL = np.arange(12) + 498 + 377 + 71 

DISORDER_INDICES_FULL = ['anxiety'] * 498 + ['depression'] * 377 + ['schizophrenia'] * 71 + ['suicidal'] * 12

def get_disorder_name(session_id,full=False):
    if full:
        anxi, depr, schi, suic = ANXI_INDICES_FULL, DEPR_INDICES_FULL, SCHI_INDICES_FULL, SUIC_INDICES_FULL
    else:
        anxi, depr, schi, suic = ANXI_INDICES, DEPR_INDICES, SCHI_INDICES, SUIC_INDICES
        
    if session_id in anxi:
        disorder_name = 'anxiety'
    elif session_id in depr:
        disorder_name = 'depression'
    elif session_id in schi:
        disorder_name = 'schizophrenia'
    elif session_id in suic:
        disorder_name = 'suicidal'
    else:
        disorder_name = 'undefined'
    return disorder_name

def get_all_vectors(emb_type='doc2vec', score_type='standard'):
    if emb_type == 'doc2vec':
        session_c_vectors = get_vectors('./doc2vec/vectors/session_c_vectors.txt')
        session_t_vectors = get_vectors('./doc2vec/vectors/session_t_vectors.txt')
    elif emb_type == 'sbert':
        session_c_vectors = np.load('sbert/vectors/session_c_vectors.npy')
        session_t_vectors = np.load('sbert/vectors/session_t_vectors.npy')

    wai_c_vectors, wai_t_vectors = get_wai_vectors(emb_type, score_type)
    return wai_c_vectors, wai_t_vectors, session_c_vectors, session_t_vectors

def get_indices(filepath,exclude=True):
    num_lines = []
    PROBLEM_SESS = [115, 163, 281, 569, 619] + [524, 804]
#     PROBLEM_SESS = []
    with open(filepath) as fp:
        for i, line in enumerate(fp):
            if exclude: 
                if i not in PROBLEM_SESS:
                    num_lines.append(int(line))
            else:
                num_lines.append(int(line))                
    indices = np.concatenate(((0,), np.cumsum(num_lines))) 
    index_pairs = [(indices[i], indices[i+1]) for i in np.arange(len(num_lines))]
    return index_pairs  

def get_all_indices(full=False):
    if full:
        indices_c = get_indices('data/alexstreet/patient/lines.list',exclude=False)
        indices_t = get_indices('data/alexstreet/therapist/lines.list',exclude=False)
    else:
        indices_c = get_indices('data/alexstreet/patient/lines.list')
        indices_t = get_indices('data/alexstreet/therapist/lines.list')
    return indices_c, indices_t

def get_index_lists(index_pairs, full=False):
    turns, sessions, disorders  = [], [], []
    for i, p in enumerate(index_pairs):
        turns += list(np.arange(0,p[1]-p[0]))
        sessions += [i] * (p[1]-p[0])
        disorders += [get_disorder_name(i,full)] * (p[1]-p[0])
    return turns, sessions, disorders

def get_all_sessions(emb_type, session_c_vectors, session_t_vectors, indices_c, indices_t, mode='compute'):
    if mode == 'load':
        sessions_emb_c = np.save(f'{emb_type}/sessions/sessions_emb_c.npy', allow_pickle=True)
        sessions_emb_t = np.save(f'{emb_type}/sessions/sessions_emb_t.npy', allow_pickle=True)
    else:
        sessions_emb_c = get_sessions(session_c_vectors, indices_c)
        sessions_emb_t = get_sessions(session_t_vectors, indices_t)
        if mode == 'save':
            np.save(f'{emb_type}/sessions/sessions_emb_c.npy', sessions_emb_c)
            np.save(f'{emb_type}/sessions/sessions_emb_t.npy', sessions_emb_t)
    return sessions_emb_c, sessions_emb_t

def get_disorder(score_list, indices):
    return np.array(score_list)[indices]
  
def get_df_dict_from_computed(scores, scales, indices, turn_owner, full=False):

    turns, sessions, disorders = get_index_lists(indices, full)
    full_scale, task_scale, bond_scale, goal_scale = scales

    data = {'turn': turns,
            'session': sessions,
            'disorder': disorders,
            'scores': list(scores),
            'full_scale': list(full_scale),
            'task_scale': list(task_scale),
            'bond_scale': list(bond_scale),
            'goal_scale': list(goal_scale),
            'turn_owner': [turn_owner] * len(turns)}
    return data

def get_df_dict(vectors, ref_vectors, indices, turn_owner, full=False):

    turns, sessions, disorders = get_index_lists(indices, full)
    scores = get_scores(vectors, ref_vectors)

    full_scale = get_scales(scores, scale='full')
    task_scale = get_scales(scores, scale='task')
    bond_scale = get_scales(scores, scale='bond')
    goal_scale = get_scales(scores, scale='goal')

    data = {'turn': turns,
            'session': sessions,
            'disorder': disorders,
            'scores': list(scores),
            'full_scale': list(full_scale),
            'task_scale': list(task_scale),
            'bond_scale': list(bond_scale),
            'goal_scale': list(goal_scale),
            'turn_owner': [turn_owner] * len(turns)}
    return data

def get_all_dtw(emb_type, session_scales_c, session_scales_t, mode='compute',prefix=''):
    
    if mode == 'load':
        d_all_c = np.load(f'{emb_type}/intermediates/{prefix}d_all_c.npy', allow_pickle=True)
        d_all_t = np.load(f'{emb_type}/intermediates/{prefix}d_all_t.npy', allow_pickle=True)
        d_full_c = np.load(f'{emb_type}/intermediates/{prefix}d_full_c.npy', allow_pickle=True)
        d_full_t = np.load(f'{emb_type}/intermediates/{prefix}d_full_t.npy', allow_pickle=True)
        d_3scales_c = np.load(f'{emb_type}/intermediates/{prefix}d_3scales_c.npy', allow_pickle=True)
        d_3scales_t = np.load(f'{emb_type}/intermediates/{prefix}d_3scales_t.npy', allow_pickle=True)
    else:
        sessions_all_c, sessions_full_scale_c, sessions_task_scale_c, sessions_bond_scale_c, sessions_goal_scale_c = session_scales_c
        sessions_all_t, sessions_full_scale_t, sessions_task_scale_t, sessions_bond_scale_t, sessions_goal_scale_t = session_scales_t
    
        sessions_3scales_t = get_3scales(sessions_task_scale_t, sessions_bond_scale_t, sessions_goal_scale_t)
        sessions_3scales_c = get_3scales(sessions_task_scale_c, sessions_bond_scale_c, sessions_goal_scale_c)
    
        d_all_c = get_dtw(sessions_all_c)
        d_all_t = get_dtw(sessions_all_t)
        d_full_c = get_dtw(sessions_full_scale_c)
        d_full_t = get_dtw(sessions_full_scale_t)
        d_3scales_c = get_dtw(sessions_3scales_c)
        d_3scales_t = get_dtw(sessions_3scales_t)
    
        if mode == 'save':
            np.save(f'{emb_type}/intermediates/{prefix}d_all_c.npy',d_all_c)
            np.save(f'{emb_type}/intermediates/{prefix}d_all_t.npy',d_all_t)
            np.save(f'{emb_type}/intermediates/{prefix}d_full_c.npy',d_full_c)
            np.save(f'{emb_type}/intermediates/{prefix}d_full_t.npy',d_full_t)
            np.save(f'{emb_type}/intermediates/{prefix}d_3scales_c.npy',d_3scales_c)
            np.save(f'{emb_type}/intermediates/{prefix}d_3scales_t.npy',d_3scales_t)

    return d_all_c, d_full_c, d_3scales_c, d_all_t, d_full_t, d_3scales_t

def plot_dr(d_c, d_t, dr_type='mds'):
    
    if dr_type == 'mds':
        
        dr = MDS(n_components=2, dissimilarity='precomputed')
        Xdr_c = dr.fit_transform(d_c)

        dr = MDS(n_components=2, dissimilarity='precomputed')
        Xdr_t = dr.fit_transform(d_t)
        
    elif dr_type == 'tsne':
        
        dr = TSNE(n_components=2, metric='precomputed')
        Xdr_c = dr.fit_transform(d_c)
        
        dr = TSNE(n_components=2, metric='precomputed')
        Xdr_t = dr.fit_transform(d_t)
        
    elif dr_type == 'isomap':
        
        dr = Isomap(n_components=2, metric='precomputed')
        Xdr_c = dr.fit_transform(d_c)
        
        dr = Isomap(n_components=2, metric='precomputed')
        Xdr_t = dr.fit_transform(d_t)
        
    sns.set(font_scale = 1.5)
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    fig.suptitle('Dimension reduction of the alliance scores')
    
    sns.set_style('whitegrid')
    plot_df = pd.DataFrame(data={f'{dr_type}_0': Xdr_c[:,0], f'{dr_type}_1': Xdr_c[:,1], 'disorder':DISORDER_INDICES})
    g = sns.scatterplot(data=plot_df, x=f'{dr_type}_0', y=f'{dr_type}_1', hue='disorder', ax=axes[0])
    axes[0].set_title('patient')
    axes[0].get_legend().remove()
    
    sns.set_style('whitegrid')
    plot_df = pd.DataFrame(data={f'{dr_type}_0': Xdr_t[:,0], f'{dr_type}_1': Xdr_t[:,1], 'disorder':DISORDER_INDICES})
    g = sns.scatterplot(data=plot_df, x=f'{dr_type}_0', y=f'{dr_type}_1', hue='disorder', ax=axes[1])
    axes[1].set_title('therapist')
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)    

def get_data_df(emb_type, scores, scales, indices, mode='compute', prefix='', full=False):
    
    scores_c, scores_t = scores
    scales_c, scales_t = scales
    indices_c, indices_t = indices

    if mode == 'load':
        data_df = pd.read_csv(f'{emb_type}/{prefix}alliance.csv')  
    else:
        dict_c = get_df_dict_from_computed(scores_c, scales_c, indices_c, 'patient', full)
        dict_t = get_df_dict_from_computed(scores_t, scales_t, indices_t, 'therapist', full)
        data_df = get_df_from_dicts(dict_c, dict_t)
    
        if mode == 'save':
            data_df.to_csv(f'{emb_type}/{prefix}alliance.csv', index=False)
    
    return data_df

def get_scale_df(data_df):
    selected_vars=["turn_owner", "turn", "disorder"]
    collapsed_vars = ["full_scale", "task_scale", "bond_scale", "goal_scale"]
    scale_df = pd.melt(data_df[selected_vars+collapsed_vars], id_vars=selected_vars, var_name='scale', value_name='score')
    return scale_df

def get_radar_df(data_df):
    
    stats = data_df.groupby(['turn_owner','disorder']).mean()
    
    dft_p = pd.DataFrame(data = {'disorder': ['anxiety', 'depression', 'schizophrenia', 'suicidal'],
                          'FULL_Scale': list(stats[:4]['full_scale']),
                          'TASK_Scale': list(stats[:4]['task_scale']),
                          'BOND_Scale': list(stats[:4]['bond_scale']),
                          'GOAL_Scale': list(stats[:4]['goal_scale']),
                          })
    dft_p.set_index('disorder', inplace=True)

    dft_t = pd.DataFrame(data = {'disorder': ['anxiety', 'depression', 'schizophrenia', 'suicidal'],
                          'FULL_Scale': list(stats[4:]['full_scale']),
                          'TASK_Scale': list(stats[4:]['task_scale']),
                          'BOND_Scale': list(stats[4:]['bond_scale']),
                          'GOAL_Scale': list(stats[4:]['goal_scale']),
                          })
    dft_t.set_index('disorder', inplace=True)
    
    radar_dfs = [dft_p, dft_t]

    return radar_dfs

def get_trajs(data_df):
    trajs = {}
    for d in ['anxiety','depression','schizophrenia','suicidal']:
        trajs[d] = {}
        for t in ['patient','therapist']:
            trajs[d][t] = data_df[(data_df['disorder']==d) & (data_df['turn_owner']==t)].groupby(['turn']).mean()
    return trajs

def print_stats(data_df):
    stats = data_df.groupby(['turn_owner','disorder']).mean()
    print('mean')
    print(stats)
    print('std')
    print(data_df.groupby(['turn_owner','disorder']).std())


def test_hypothesis(data_df, hypothesis):
    
    if hypothesis == 'suicidality':
        
        print('========= t-Test for suicidality =========')
        group1 = data_df[data_df["disorder"] == "suicidal"]
        group2 = data_df[data_df["disorder"] != "suicidal"]
        for scale in ["full_scale","task_scale","bond_scale","goal_scale"]:
            x1 = group1[scale]
            x2 = group2[scale]
            print(scale, stats.ttest_ind(x1, x2))
            
    elif hypothesis == 'turn_owner':
        
        print('========= t-Test for turn owner =========')
        group1 = data_df[data_df["turn_owner"] == "patient"]
        group2 = data_df[data_df["turn_owner"] == "therapist"]

        for scale in ["full_scale","task_scale","bond_scale","goal_scale"]:
            x1 = group1[scale]
            x2 = group2[scale]
            print(scale, stats.ttest_ind(x1, x2))
    
    elif hypothesis == 'disorder':
        
        print('========= t-Test for disorder =========')
        for d1 in ["anxiety","depression","schizophrenia","suicidal"]:
            for d2 in ["anxiety","depression","schizophrenia","suicidal"]:
                group1 = data_df[data_df["disorder"] == d1]
                group2 = data_df[data_df["disorder"] == d2]
                print("=====", d1, d2, "=====")
                for scale in ["full_scale","task_scale","bond_scale","goal_scale"]:
                    x1 = group1[scale]
                    x2 = group2[scale]
                    print(scale, stats.ttest_ind(x1, x2))
    

def plot_analytics(data, plot_type):
    
    if plot_type == 'relation':
        
        data_df = data
        
        sns.set_theme(style="ticks")
        selected_vars=["task_scale","bond_scale","goal_scale", "disorder"]
        g = sns.pairplot(data_df[data_df['turn_owner']=='patient'][selected_vars], hue="disorder")

        sns.set_theme(style="ticks")
        selected_vars=["task_scale","bond_scale","goal_scale", "disorder"]
        g = sns.pairplot(data_df[data_df['turn_owner']=='therapist'][selected_vars], hue="disorder")

    elif plot_type == 'suicidality':
        
        scale_df = data
        
        suicidality = np.array(scale_df["disorder"] == "suicidal")
        suicidality_list = np.array(['non-suicidal'] * len(suicidality))
        suicidality_list[suicidality] = 'suicidal'
        scale_df['suicidality'] = suicidality_list

        sns.set_theme(style="ticks")
        sns.set(font_scale = 1.5)

        fig, ax = plt.subplots(figsize=(10, 10))
        fig.suptitle('Suicidality in four scales')

        hue_parameters= {'data': scale_df, 'x': 'scale','y':'score', 'hue': 'suicidality'}
        g = sns.boxplot(**hue_parameters, ax=ax)
        ax.set(xlabel=None)
        
        pairs=[[("full_scale", "suicidal"), ("full_scale", "non-suicidal")],
              [("task_scale", "suicidal"), ("task_scale", "non-suicidal")],
              [("bond_scale", "suicidal"), ("bond_scale", "non-suicidal")],
              [("goal_scale", "suicidal"), ("goal_scale", "non-suicidal")]]
        annotator = Annotator(ax, pairs, **hue_parameters)
        annotator.configure(test='t-test_ind', text_format='star', loc='inside')
        annotator.apply_and_annotate()
        
    elif plot_type == 'dynamics':
        
        scale_df = data

        sns.set_style('whitegrid')
        conditions = [x in np.arange(0,101,10) for x in scale_df["turn"]]

        sns.set(font_scale = 1.5)
        g = sns.catplot(x="turn", y="score", hue="turn_owner", col="disorder", row="scale",
                        capsize=.2, height=6, aspect=1., sharey='row',
                        kind="point", data=scale_df[conditions])
        
    elif plot_type == 'dynamics_fine':
        
        scale_df = data
        
        conditions = [x in np.arange(0,101,10) for x in scale_df["turn"]]
        conditions = scale_df["turn"] % 5 == 0

        sns.set(font_scale = 1.5)
        g = sns.relplot(
            data=scale_df[conditions],
            x="turn", y="score", hue="turn_owner", col="disorder", row="scale",
            kind="line",
            height=5, aspect=1, facet_kws=dict(sharex=False, sharey='row'),
        )

    elif plot_type == 'regression':

        data_df = data
        
        sns.set_style('whitegrid')
        sns.set(font_scale = 1.5)
        g = sns.lmplot(x="turn", y="full_scale", hue="turn_owner", col="disorder", data=data_df, 
                   sharex=False, scatter_kws = {"alpha": 0.01, "s":50}, line_kws={'lw': 3});
        for lh in g._legend.legendHandles: 
            lh.set_alpha(1)
            lh._sizes = [50] 
    
    elif plot_type == 'regression_scale':
        
        scale_df = data

        sns.set_style('whitegrid')
        sns.set(font_scale = 1.3)
        conditions = [True] * len(scale_df["turn"])
        g = sns.lmplot(x="turn", y="score", hue="turn_owner", col="disorder", row="scale",
                        sharex='col', sharey='row',
                       scatter_kws = {"alpha": 0.01, "s":50}, line_kws={'lw': 3}, data=scale_df[conditions])
        
    elif plot_type == 'regression_turn_owner':
        
        data_df = data
        
        sns.set_style('whitegrid')
        g = sns.lmplot(x="turn", y="full_scale", hue="disorder", col="turn_owner", data=data_df, 
                   sharex=False, scatter_kws = {"alpha": 0.01, "s":1}, line_kws={'lw': 2});
        for lh in g._legend.legendHandles: 
            lh.set_alpha(1)
            lh._sizes = [5] 
            
    elif plot_type == 'violin_disorder':
        
        data_df = data
        
        sns.set_theme(style="ticks")
        sns.set(font_scale = 1.5)

        fig, axes = plt.subplots(2, 2, figsize=(15, 15))
        fig.suptitle('Alliance score ranges in four scales')

        g = sns.violinplot(x="disorder", y="full_scale", ax=axes[0][0],
                    hue="turn_owner",
                    data=data_df)
        axes[0][0].set_title('full_scale')
        axes[0][0].set(xlabel=None)

        g = sns.violinplot(x="disorder", y="task_scale", ax=axes[0][1],
                    hue="turn_owner",
                    data=data_df)
        axes[0][1].set_title('task_scale')
        axes[0][1].set(xlabel=None)

        g = sns.violinplot(x="disorder", y="bond_scale", ax=axes[1][0],
                    hue="turn_owner",
                    data=data_df)
        axes[1][0].set_title('bond_scale')
        axes[1][0].set(xlabel=None)

        g = sns.violinplot(x="disorder", y="goal_scale", ax=axes[1][1],
                    hue="turn_owner",
                    data=data_df)
        axes[1][1].set_title('goal_scale')
        axes[1][1].set(xlabel=None)

            
    elif plot_type == 'boxplot_disorder':
        
        data_df = data
        
        sns.set_theme(style="ticks")
        sns.set(font_scale = 1.5)

        fig, axes = plt.subplots(2, 2, figsize=(15, 15))
        fig.suptitle('Alliance score ranges in four scales')

        g = sns.boxplot(x="disorder", y="full_scale", ax=axes[0][0],
                    hue="turn_owner",
                    data=data_df)
        axes[0][0].set_title('full_scale')
        axes[0][0].set(xlabel=None)

        g = sns.boxplot(x="disorder", y="task_scale", ax=axes[0][1],
                    hue="turn_owner",
                    data=data_df)
        axes[0][1].set_title('task_scale')
        axes[0][1].set(xlabel=None)

        g = sns.boxplot(x="disorder", y="bond_scale", ax=axes[1][0],
                    hue="turn_owner",
                    data=data_df)
        axes[1][0].set_title('bond_scale')
        axes[1][0].set(xlabel=None)

        g = sns.boxplot(x="disorder", y="goal_scale", ax=axes[1][1],
                    hue="turn_owner",
                    data=data_df)
        axes[1][1].set_title('goal_scale')
        axes[1][1].set(xlabel=None)

    elif plot_type == 'boxplot_annotate1':
        
        data_df = data

        sns.set_theme(style="ticks")
        sns.set(font_scale = 1.4)

        fig, axes = plt.subplots(2, 2, figsize=(15, 15))
        fig.suptitle('Alliance score ranges in four scales')

        for i, scale in enumerate(["full_scale","task_scale","bond_scale","goal_scale"]):
            a, b = int(i / 2), i % 2
            hue_parameters= {'data': data_df,'x': 'disorder','y': scale,'hue': "turn_owner"}
            g = sns.boxplot(**hue_parameters, ax=axes[a][b])
            axes[a][b].set_title(scale)
            axes[a][b].set(xlabel=None)
            if i != 0:
                axes[a][b].get_legend().remove()

            pairs1=[[("anxiety", "therapist"), ("anxiety", "patient")],
                  [("depression", "therapist"), ("depression", "patient")],
                  [("schizophrenia", "therapist"), ("schizophrenia", "patient")],
                  [("suicidal", "therapist"), ("suicidal", "patient")]]
            pairs2=[[("anxiety", "therapist"), ("depression", "therapist")],
                    [("anxiety", "patient"), ("depression", "patient")],
                    [("anxiety", "therapist"), ("schizophrenia", "therapist")],
                    [("anxiety", "patient"), ("schizophrenia", "patient")],
                    [("anxiety", "therapist"), ("suicidal", "therapist")],
                    [("anxiety", "patient"), ("suicidal", "patient")],
                    [("depression", "therapist"), ("schizophrenia", "therapist")],
                    [("depression", "patient"), ("schizophrenia", "patient")],
                    [("depression", "therapist"), ("suicidal", "therapist")],
                    [("depression", "patient"), ("suicidal", "patient")],
                    [("schizophrenia", "therapist"), ("suicidal", "therapist")],
                    [("schizophrenia", "patient"), ("suicidal", "patient")]]
            pairs = pairs1
            annotator = Annotator(axes[a][b], pairs, **hue_parameters)
            annotator.configure(test='t-test_ind', text_format='star', loc='inside')
            annotator.apply_and_annotate()
            

    elif plot_type == 'boxplot_annotate2':
        
        data_df = data

        sns.set_theme(style="ticks")
        sns.set(font_scale = 1.4)

        fig, axes = plt.subplots(2, 2, figsize=(15, 15))
        fig.suptitle('Alliance score ranges in four scales')

        for i, scale in enumerate(["full_scale","task_scale","bond_scale","goal_scale"]):
            a, b = int(i / 2), i % 2
            hue_parameters= {'data': data_df,'x': 'disorder','y': scale,'hue': "turn_owner"}
            g = sns.boxplot(**hue_parameters, ax=axes[a][b])
            axes[a][b].set_title(scale)
            axes[a][b].set(xlabel=None)
            if i == 3:
                axes[a][b].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
            else:
                axes[a][b].get_legend().remove()
        
            pairs1=[[("anxiety", "therapist"), ("anxiety", "patient")],
                  [("depression", "therapist"), ("depression", "patient")],
                  [("schizophrenia", "therapist"), ("schizophrenia", "patient")],
                  [("suicidal", "therapist"), ("suicidal", "patient")]]
            pairs2=[[("anxiety", "therapist"), ("depression", "therapist")],
                    [("anxiety", "patient"), ("depression", "patient")],
                    [("anxiety", "therapist"), ("schizophrenia", "therapist")],
                    [("anxiety", "patient"), ("schizophrenia", "patient")],
                    [("anxiety", "therapist"), ("suicidal", "therapist")],
                    [("anxiety", "patient"), ("suicidal", "patient")],
                    [("depression", "therapist"), ("schizophrenia", "therapist")],
                    [("depression", "patient"), ("schizophrenia", "patient")],
                    [("depression", "therapist"), ("suicidal", "therapist")],
                    [("depression", "patient"), ("suicidal", "patient")],
                    [("schizophrenia", "therapist"), ("suicidal", "therapist")],
                    [("schizophrenia", "patient"), ("suicidal", "patient")]]
            pairs = pairs2
            annotator = Annotator(axes[a][b], pairs, **hue_parameters)
            annotator.configure(test='t-test_ind', text_format='star', loc='inside')
            annotator.apply_and_annotate()      

    elif plot_type == 'boxplot_annotate3':
        
        data_df = data
        
        sns.set_theme(style="ticks")
        sns.set(font_scale = 1.4)

        fig, axes = plt.subplots(2, 2, figsize=(15, 15))
        fig.suptitle('Alliance score ranges in four scales')

        for i, scale in enumerate(["full_scale","task_scale","bond_scale","goal_scale"]):
            a, b = int(i / 2), i % 2
            hue_parameters= {'data': data_df,'x': 'disorder','y': scale}
            g = sns.boxplot(**hue_parameters, ax=axes[a][b])
            axes[a][b].set_title(scale)
            axes[a][b].set(xlabel=None)
        
            pairs=[("anxiety", "depression"), ("anxiety", "suicidal"), ("anxiety", "schizophrenia"),
                   ("depression", "schizophrenia"), ("depression", "suicidal"),("schizophrenia", "suicidal")]
            annotator = Annotator(axes[a][b], pairs, **hue_parameters)
            annotator.configure(test='t-test_ind', text_format='star', loc='inside')
            annotator.apply_and_annotate()
        
    elif plot_type == 'radar_disorder':
        
        radar_dfs = data
        dft_p, dft_t = radar_dfs        
                
        def plot_dft_radar(dft, title):

            sns.set_style("white")
            dft=(dft-dft.mean())/dft.std()
            # Each attribute we'll plot in the radar chart.
            labels = ['FULL_Scale','TASK_Scale', 'BOND_Scale', 'GOAL_Scale']

            # Number of variables we're plotting.
            num_vars = len(labels)

            # Split the circle into even parts and save the angles
            # so we know where to put each axis.
            angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

            # The plot is a circle, so we need to "complete the loop"
            # and append the start value to the end.
            angles += angles[:1]

            # ax = plt.subplot(polar=True)
            fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

            # Helper function to plot each disorder on the radar chart.
            def add_to_radar(disorder, color):
                values = dft.loc[disorder].tolist()
                values += values[:1]
                ax.plot(angles, values, color=color, linewidth=1, label=disorder)
                ax.fill(angles, values, color=color, alpha=0.2)

            # Add each car to the chart.
            add_to_radar('anxiety', '#1aaf6c')
            add_to_radar('depression', '#429bf4')
            add_to_radar('schizophrenia', '#d42cea')
            add_to_radar('suicidal', '#dc7633')

            # Fix axis to go in the right order and start at 12 o'clock.
            ax.set_theta_offset(np.pi / 2)
            ax.set_theta_direction(-1)

            # Draw axis lines for each angle and label.
            ax.set_thetagrids(np.degrees(angles)[:-1], labels)

            # Go through labels and adjust alignment based on where
            # it is in the circle.
            for label, angle in zip(ax.get_xticklabels(), angles):
                if angle in (0, np.pi):
                    label.set_horizontalalignment('center')
                elif 0 < angle < np.pi:
                    label.set_horizontalalignment('left')
                else:
                    label.set_horizontalalignment('right')

            # Ensure radar goes from 0 to 100.
            # ax.set_ylim(0, 100)
            # You can also set gridlines manually like this:
            # ax.set_rgrids([20, 40, 60, 80, 100])

            # Set position of y-labels (0-100) to be in the middle
            # of the first two axes.
            ax.set_rlabel_position(180 / num_vars)

            # Add some custom styling.
            # Change the color of the tick labels.
            ax.tick_params(colors='#222222')
            # Make the y-axis (0-100) labels smaller.
            ax.tick_params(axis='y', labelsize=8)
            # Change the color of the circular gridlines.
            ax.grid(color='#AAAAAA')
            # Change the color of the outermost gridline (the spine).
            ax.spines['polar'].set_color('#222222')
            # Change the background color inside the circle itself.
            ax.set_facecolor('#FAFAFA')

            # Add title.
            ax.set_title(title, y=1.08)

            # Add a legend as well.
            ax.legend(loc='upper right', bbox_to_anchor=(1.6, 1))

        plot_dft_radar(dft_p, 'Comparing conditions across WAI scales (Patient)')
        plot_dft_radar(dft_t, 'Comparing conditions across WAI scales (Therapist)')

    elif plot_type == 'trajectory':
        
        trajs = data

        sns.set(font_scale = 2)
        plt.style.use('classic')
        fig = plt.figure(figsize=(10,10))
        ax = plt.axes(projection='3d')
        fig.patch.set_facecolor('white')
        colors = ['purple','blue','green','orange']
        styles = ['-','--']
        cmaps = ['Purples', 'Blues', 'Greens', 'Oranges']

        timesteps = np.arange(0,101,25)
        x1,x2,x3 = 'bond_scale','task_scale','goal_scale'

        for j,d in enumerate(['anxiety','depression','schizophrenia','suicidal']):
            for i, t in enumerate(['patient','therapist']):
                plot_df = trajs[d][t]
                x,y,z = plot_df[x1][timesteps], plot_df[x2][timesteps], plot_df[x3][timesteps]
        
                timesteps_ = np.linspace(timesteps.min(), timesteps.max(), 100)
                spline = make_interp_spline(timesteps, x)
                x_ = spline(timesteps_)
                spline = make_interp_spline(timesteps, y)
                y_ = spline(timesteps_)
                spline = make_interp_spline(timesteps, z)
                z_ = spline(timesteps_)
        
                ax.plot3D(x_,y_,z_, color=colors[j],linestyle=styles[i], linewidth=3, label=str(d)+' '+str(t))
                ax.scatter3D(x[-1:], y[-1:], z[-1:], c=colors[j], s=timesteps[-1:]);
        ax.set_xlabel(x1)
        ax.set_ylabel(x2)
        ax.set_zlabel(x3)
        ax.legend()
        ax.xaxis.labelpad=10
        ax.yaxis.labelpad=10
        ax.zaxis.labelpad=10
        fig.tight_layout()

