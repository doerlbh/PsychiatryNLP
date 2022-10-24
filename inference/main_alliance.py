#!/usr/bin/env python
# coding: utf-8

from alexstreet_utils import *

emb_type = 'doc2vec'
score_type = 'standard'
prefix = ''

print('>>> inference step')

indices_c, indices_t = get_all_indices()
wai_c_vectors, wai_t_vectors, session_c_vectors, session_t_vectors = get_all_vectors(emb_type, score_type)
sessions_emb_c, sessions_emb_t = get_all_sessions(emb_type, session_c_vectors, session_t_vectors, indices_c, indices_t)
indices = list(get_all_indices(True))

scales_c, sessions_c, scores_c = get_all_scales_and_sessions(emb_type, 'c', session_c_vectors, wai_c_vectors, indices_c, score_type, mode='compute', prefix=prefix)
full_scale_c, task_scale_c, bond_scale_c, goal_scale_c = scales_c
sessions_all_c, sessions_full_scale_c, sessions_task_scale_c, sessions_bond_scale_c, sessions_goal_scale_c = sessions_c

scales_t, sessions_t, scores_t = get_all_scales_and_sessions(emb_type, 't', session_t_vectors, wai_t_vectors, indices_t, score_type, mode='compute', prefix=prefix)
full_scale_t, task_scale_t, bond_scale_t, goal_scale_t = scales_t
sessions_all_t, sessions_full_scale_t, sessions_task_scale_t, sessions_bond_scale_t, sessions_goal_scale_t = sessions_t

scores = [scores_c, scores_t]
scales = [scales_c, scales_t]

# dimension reduction (optional)

print('>>> dimension reduction step')

d_all_c, d_full_c, d_3scales_c, d_all_t, d_full_t, d_3scales_t = get_all_dtw(emb_type, sessions_c, sessions_t, mode='compute',prefix=prefix)

plot_dr(d_3scales_c, d_3scales_t, dr_type='mds')
plot_dr(d_3scales_c, d_3scales_t, dr_type='tsne')
plot_dr(d_3scales_c, d_3scales_t, dr_type='isomap')

# collate data

print('>>> data collation step')

data_df = get_data_df(emb_type, scores, scales, indices, mode='compute', prefix=prefix)
scale_df = get_scale_df(data_df)
radar_dfs = get_radar_df(data_df)
trajs = get_trajs(data_df)
    
# hypothesis testing

print('>>> hypothesis testing step')

test_hypothesis(data_df, 'suidicality')
test_hypothesis(data_df, 'turn_owner')
test_hypothesis(data_df, 'disorder')

# analytics

print('>>> visual analytics step')

print_stats(data_df)

plot_analytics(data_df, 'relation')
plot_analytics(scale_df, 'suicidality')
plot_analytics(scale_df, 'dynamics')
plot_analytics(data_df, 'regression')
plot_analytics(data_df, 'boxplot_disorder')
plot_analytics(data_df, 'boxplot_annotate')
plot_analytics(data_df, 'boxplot_annotate1')
plot_analytics(data_df, 'boxplot_annotate2')
plot_analytics(data_df, 'boxplot_annotate3')
plot_analytics(trajs, 'trajectory')
plot_analytics(radar_dfs, 'radar_disorder')

# other analytics (optional)

plot_analytics(scale_df, 'dynamics_fine')
plot_analytics(data_df, 'regression_scale')
plot_analytics(data_df, 'regression_turn_owner')
plot_analytics(data_df, 'violin_disorder')

