#!/usr/bin/env python
# coding: utf-8


from classify_utils import *

get_ipython().run_line_magic('matplotlib', 'inline')

max_iter = 30000
selected_ckpt = 15000
n_test_sample = 10000


feature_type='waiemb'
embedding_type='sbert'
turn_owner='c'

train_data, test_data, feature_dim = run_construct_data(embedding_type, feature_type, turn_owner)

for model_type in ['transformer', 'lstm', 'rnn']:
    model, ckpt_path, fig_path = run_construct_model(embedding_type, feature_type, turn_owner, model_type, feature_dim)
    model = run_train(model, model_type, ckpt_path, train_data, test_data, max_iter=max_iter)
    accuracy, confusion_accuracy, confusion = run_test(model, model_type, train_data, test_data, ckpt_path, 
                                                   n_test_sample=n_test_sample, selected_ckpt=selected_ckpt)
    print(feature_type, embedding_type, turn_owner, model_type, accuracy)
    name = f'classify/figures/confusion_{model_type}_{embedding_type}_{feature_type}_{turn_owner}.png'
    plot_confusion(confusion, name)


feature_type='wai'
embedding_type='sbert'
turn_owner='c'

train_data, test_data, feature_dim = run_construct_data(embedding_type, feature_type, turn_owner)

for model_type in ['transformer', 'lstm', 'rnn']:
    model, ckpt_path, fig_path = run_construct_model(embedding_type, feature_type, turn_owner, model_type, feature_dim)
    model = run_train(model, model_type, ckpt_path, train_data, test_data, max_iter=max_iter)
    accuracy, confusion_accuracy, confusion = run_test(model, model_type, train_data, test_data, ckpt_path, 
                                                   n_test_sample=n_test_sample, selected_ckpt=selected_ckpt)
    print(feature_type, embedding_type, turn_owner, model_type, accuracy)
    name = f'classify/figures/confusion_{model_type}_{embedding_type}_{feature_type}_{turn_owner}.png'
    plot_confusion(confusion, name)


feature_type='emb'
embedding_type='sbert'
turn_owner='c'

train_data, test_data, feature_dim = run_construct_data(embedding_type, feature_type, turn_owner)

for model_type in ['transformer', 'lstm', 'rnn']:
    model, ckpt_path, fig_path = run_construct_model(embedding_type, feature_type, turn_owner, model_type, feature_dim)
    model = run_train(model, model_type, ckpt_path, train_data, test_data, max_iter=max_iter)
    accuracy, confusion_accuracy, confusion = run_test(model, model_type, train_data, test_data, ckpt_path, 
                                                   n_test_sample=n_test_sample, selected_ckpt=selected_ckpt)
    print(feature_type, embedding_type, turn_owner, model_type, accuracy)
    name = f'classify/figures/confusion_{model_type}_{embedding_type}_{feature_type}_{turn_owner}.png'
    plot_confusion(confusion, name)


feature_type='waiemb'
embedding_type='sbert'
turn_owner='t'

train_data, test_data, feature_dim = run_construct_data(embedding_type, feature_type, turn_owner)

for model_type in ['transformer', 'lstm', 'rnn']:
    model, ckpt_path, fig_path = run_construct_model(embedding_type, feature_type, turn_owner, model_type, feature_dim)
    model = run_train(model, model_type, ckpt_path, train_data, test_data, max_iter=max_iter)
    accuracy, confusion_accuracy, confusion = run_test(model, model_type, train_data, test_data, ckpt_path, 
                                                   n_test_sample=n_test_sample, selected_ckpt=selected_ckpt)
    print(feature_type, embedding_type, turn_owner, model_type, accuracy)
    name = f'classify/figures/confusion_{model_type}_{embedding_type}_{feature_type}_{turn_owner}.png'
    plot_confusion(confusion, name)


feature_type='wai'
embedding_type='sbert'
turn_owner='t'

train_data, test_data, feature_dim = run_construct_data(embedding_type, feature_type, turn_owner)

for model_type in ['transformer', 'lstm', 'rnn']:
    model, ckpt_path, fig_path = run_construct_model(embedding_type, feature_type, turn_owner, model_type, feature_dim)
    model = run_train(model, model_type, ckpt_path, train_data, test_data, max_iter=max_iter)
    accuracy, confusion_accuracy, confusion = run_test(model, model_type, train_data, test_data, ckpt_path, 
                                                   n_test_sample=n_test_sample, selected_ckpt=selected_ckpt)
    print(feature_type, embedding_type, turn_owner, model_type, accuracy)
    name = f'classify/figures/confusion_{model_type}_{embedding_type}_{feature_type}_{turn_owner}.png'
    plot_confusion(confusion, name)


feature_type='emb'
embedding_type='sbert'
turn_owner='t'

train_data, test_data, feature_dim = run_construct_data(embedding_type, feature_type, turn_owner)

for model_type in ['transformer', 'lstm', 'rnn']:
    model, ckpt_path, fig_path = run_construct_model(embedding_type, feature_type, turn_owner, model_type, feature_dim)
    model = run_train(model, model_type, ckpt_path, train_data, test_data, max_iter=max_iter)
    accuracy, confusion_accuracy, confusion = run_test(model, model_type, train_data, test_data, ckpt_path, 
                                                   n_test_sample=n_test_sample, selected_ckpt=selected_ckpt)
    print(feature_type, embedding_type, turn_owner, model_type, accuracy)
    name = f'classify/figures/confusion_{model_type}_{embedding_type}_{feature_type}_{turn_owner}.png'
    plot_confusion(confusion, name)


feature_type='waiemb'
embedding_type='doc2vec'
turn_owner='c'

train_data, test_data, feature_dim = run_construct_data(embedding_type, feature_type, turn_owner)

for model_type in ['transformer', 'lstm', 'rnn']:
    model, ckpt_path, fig_path = run_construct_model(embedding_type, feature_type, turn_owner, model_type, feature_dim)
    model = run_train(model, model_type, ckpt_path, train_data, test_data, max_iter=max_iter)
    accuracy, confusion_accuracy, confusion = run_test(model, model_type, train_data, test_data, ckpt_path, 
                                                   n_test_sample=n_test_sample, selected_ckpt=selected_ckpt)
    print(feature_type, embedding_type, turn_owner, model_type, accuracy)
    name = f'classify/figures/confusion_{model_type}_{embedding_type}_{feature_type}_{turn_owner}.png'
    plot_confusion(confusion, name)



feature_type='wai'
embedding_type='doc2vec'
turn_owner='c'

train_data, test_data, feature_dim = run_construct_data(embedding_type, feature_type, turn_owner)

for model_type in ['transformer', 'lstm', 'rnn']:
    model, ckpt_path, fig_path = run_construct_model(embedding_type, feature_type, turn_owner, model_type, feature_dim)
    model = run_train(model, model_type, ckpt_path, train_data, test_data, max_iter=max_iter)
    accuracy, confusion_accuracy, confusion = run_test(model, model_type, train_data, test_data, ckpt_path, 
                                                   n_test_sample=n_test_sample, selected_ckpt=selected_ckpt)
    print(feature_type, embedding_type, turn_owner, model_type, accuracy)
    name = f'classify/figures/confusion_{model_type}_{embedding_type}_{feature_type}_{turn_owner}.png'
    plot_confusion(confusion, name)


feature_type='emb'
embedding_type='doc2vec'
turn_owner='c'

train_data, test_data, feature_dim = run_construct_data(embedding_type, feature_type, turn_owner)

for model_type in ['transformer', 'lstm', 'rnn']:
    model, ckpt_path, fig_path = run_construct_model(embedding_type, feature_type, turn_owner, model_type, feature_dim)
    model = run_train(model, model_type, ckpt_path, train_data, test_data, max_iter=max_iter)
    accuracy, confusion_accuracy, confusion = run_test(model, model_type, train_data, test_data, ckpt_path, 
                                                   n_test_sample=n_test_sample, selected_ckpt=selected_ckpt)
    print(feature_type, embedding_type, turn_owner, model_type, accuracy)
    name = f'classify/figures/confusion_{model_type}_{embedding_type}_{feature_type}_{turn_owner}.png'
    plot_confusion(confusion, name)



feature_type='waiemb'
embedding_type='doc2vec'
turn_owner='t'

train_data, test_data, feature_dim = run_construct_data(embedding_type, feature_type, turn_owner)

for model_type in ['transformer', 'lstm', 'rnn']:
    model, ckpt_path, fig_path = run_construct_model(embedding_type, feature_type, turn_owner, model_type, feature_dim)
    model = run_train(model, model_type, ckpt_path, train_data, test_data, max_iter=max_iter)
    accuracy, confusion_accuracy, confusion = run_test(model, model_type, train_data, test_data, ckpt_path, 
                                                   n_test_sample=n_test_sample, selected_ckpt=selected_ckpt)
    print(feature_type, embedding_type, turn_owner, model_type, accuracy)
    name = f'classify/figures/confusion_{model_type}_{embedding_type}_{feature_type}_{turn_owner}.png'
    plot_confusion(confusion, name)


feature_type='wai'
embedding_type='doc2vec'
turn_owner='t'

train_data, test_data, feature_dim = run_construct_data(embedding_type, feature_type, turn_owner)

for model_type in ['transformer', 'lstm', 'rnn']:
    model, ckpt_path, fig_path = run_construct_model(embedding_type, feature_type, turn_owner, model_type, feature_dim)
    model = run_train(model, model_type, ckpt_path, train_data, test_data, max_iter=max_iter)
    accuracy, confusion_accuracy, confusion = run_test(model, model_type, train_data, test_data, ckpt_path, 
                                                   n_test_sample=n_test_sample, selected_ckpt=selected_ckpt)
    print(feature_type, embedding_type, turn_owner, model_type, accuracy)
    name = f'classify/figures/confusion_{model_type}_{embedding_type}_{feature_type}_{turn_owner}.png'
    plot_confusion(confusion, name)


feature_type='emb'
embedding_type='doc2vec'
turn_owner='t'

train_data, test_data, feature_dim = run_construct_data(embedding_type, feature_type, turn_owner)

for model_type in ['transformer', 'lstm', 'rnn']:
    model, ckpt_path, fig_path = run_construct_model(embedding_type, feature_type, turn_owner, model_type, feature_dim)
    model = run_train(model, model_type, ckpt_path, train_data, test_data, max_iter=max_iter)
    accuracy, confusion_accuracy, confusion = run_test(model, model_type, train_data, test_data, ckpt_path, 
                                                   n_test_sample=n_test_sample, selected_ckpt=selected_ckpt)
    print(feature_type, embedding_type, turn_owner, model_type, accuracy)
    name = f'classify/figures/confusion_{model_type}_{embedding_type}_{feature_type}_{turn_owner}.png'
    plot_confusion(confusion, name)

