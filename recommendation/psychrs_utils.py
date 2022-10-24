import pandas as pd
import numpy as np
import datetime
import random
import time
import recnn
import torch

from tqdm.auto import tqdm

def prepare_psych_dataset(args_mut, kwargs):
    
    # get args
    frame_size = kwargs.get('frame_size')
    key_to_id = args_mut.base.key_to_id
    df = args_mut.df

    df['score'] = df['score']
    df['turn'] = df['turn']
    df['topic_id'] = df['topic_id'].apply(key_to_id.get)

    users = df[['subj_id', 'topic_id']].groupby(['subj_id']).size()
    users = users[users > frame_size].sort_values(ascending=False).index

    # If using modin: pandas groupby is sync and doesnt affect performance
    # if pd.get_type() == "modin": df = df._to_pandas()
    ratings = df.sort_values(by='turn').set_index('subj_id').drop('turn', axis=1).groupby('subj_id')

    # Groupby user
    user_dict = {}

    def app(x):
        userid = x.index[0]
        user_dict[int(userid)] = {}
        user_dict[int(userid)]['items'] = x['topic_id'].values
        user_dict[int(userid)]['ratings'] = x['score'].values

    ratings.apply(app)

    args_mut.user_dict = user_dict
    args_mut.users = users

    return args_mut, kwargs

