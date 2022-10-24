import numpy as np
import pandas as pd
import torch.nn as nn
import torch
import os
import random
import time
import math
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

from sklearn.model_selection import train_test_split

# global variables specific to the psychotherapy dataset

all_categories = ['anxiety','depression','schizophrenia','suicidal']
category_lines = {}
n_categories = len(all_categories)
labels = np.array(['anxiety'] * (498 - 3) + ['depression'] * (377 - 4) + ['schizophrenia'] * 71 + ['suicidal'] * 12)

def get_features(embedding_type='sbert',feature_type='wai', turn_owner='t'):
    if embedding_type == 'sbert':
        sessions_all_t = np.load('sbert/sessions/sessions_all_t.npy',allow_pickle=True)
        sessions_all_c = np.load('sbert/sessions/sessions_all_c.npy',allow_pickle=True)
        sessions_emb_t = np.load('sbert/sessions/sessions_emb_t.npy',allow_pickle=True)
        sessions_emb_c = np.load('sbert/sessions/sessions_emb_c.npy',allow_pickle=True)
        
    elif embedding_type == 'doc2vec':
        sessions_all_t = np.load('doc2vec/sessions/sessions_all_t.npy',allow_pickle=True)
        sessions_all_c = np.load('doc2vec/sessions/sessions_all_c.npy',allow_pickle=True)
        sessions_emb_t = np.load('doc2vec/sessions/sessions_emb_t.npy',allow_pickle=True)
        sessions_emb_c = np.load('doc2vec/sessions/sessions_emb_c.npy',allow_pickle=True)
    
    if feature_type=='wai':
        features_t = sessions_all_t
        features_c = sessions_all_c
    elif feature_type=='emb':
        features_t = sessions_emb_t
        features_c = sessions_emb_c
    elif feature_type=='waiemb':
        features_t = [np.concatenate((np.array(pair[0]), np.array(pair[1])),axis=1) for pair in zip(sessions_emb_t, sessions_all_t)]
        features_c = [np.concatenate((np.array(pair[0]), np.array(pair[1])),axis=1) for pair in zip(sessions_emb_c, sessions_all_c)]
        
    if turn_owner=='t':
        features = features_t
        
    elif turn_owner=='c':
        features = features_c
        
    return features
    
def lineToTensor(line):
    feature = np.array(line)
    feature = np.expand_dims(feature, axis=1)
    tensor = torch.from_numpy(feature).float()
    return tensor

def batchToTensor(lines):
    feature = np.array(line)
    feature = np.swapaxes(feature, 0, 1)
    tensor = torch.from_numpy(feature).float()
    return tensor

def categoryFromOutput(output):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i

def generate_square_subsequent_mask(sz: int):
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]

def randomTrainingExample(train_data, test_data, test=False):
    category = randomChoice(all_categories)
    if test:
        line = randomChoice(test_data[category])
    else:
        line = randomChoice(train_data[category])
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    line_tensor = lineToTensor(line)
    return category, line, category_tensor, line_tensor

def train(model, optimizer, batch_category_tensor, batch_tensor, update=True, model_type='lstm'):
    criterion = nn.NLLLoss()

    model.zero_grad()
    
    if model_type == 'transformer':
        src_mask = generate_square_subsequent_mask(batch_tensor.shape[0])
        output = model(batch_tensor, src_mask)
    else:      
        hidden = model.initHidden() 
        for i in range(batch_tensor.size()[0]):
            output, hidden = model(batch_tensor[i], hidden)

    loss = criterion(output, batch_category_tensor)
        
    if update:
        loss.backward()
        optimizer.step()       

    return model, optimizer, output, loss

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

# Just return an output given a line
def evaluate(model, line_tensor, model_type):
    
    if model_type == 'transformer':
        src_mask = generate_square_subsequent_mask(line_tensor.shape[0])
        output = model(line_tensor, src_mask)
    else:      
        hidden = model.initHidden() 
        for i in range(line_tensor.size()[0]):
            output, hidden = model(line_tensor[i], hidden)

    return output

def plot_confusion(confusion,name='test.png'):
    
    # Set up plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(confusion.numpy(),cmap=matplotlib.cm.Blues,vmin=0,vmax=1)
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + all_categories, rotation=90)
    ax.set_yticklabels([''] + all_categories)

    # Force label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    # sphinx_gallery_thumbnail_number = 2
    plt.savefig(name, dpi=300, bbox_inches='tight')
    plt.show()

def run_construct_data(embedding_type, feature_type, turn_owner):

    # train test data 
    
    used_features = get_features(embedding_type=embedding_type,feature_type=feature_type, turn_owner=turn_owner)
    feature_dim = len(used_features[0][0])
    for category in all_categories:
        category_lines[category] = np.array(used_features)[labels == category]

    # train test separation

    np.random.seed(0)
    category_lines_train, category_lines_test = {}, {}
    for category in all_categories:
        category_lines_train[category], category_lines_test[category] = train_test_split(category_lines[category],test_size=0.2)

    return category_lines_train, category_lines_test, feature_dim

def run_construct_model(embedding_type, feature_type, turn_owner, model_type, feature_dim):
    file_path = f'classify/{embedding_type}/{model_type}/{feature_type}'
    ckpt_path = f'{file_path}/checkpoints/{turn_owner}'
    fig_path = f'{file_path}/figures/{turn_owner}'
    
    Path(ckpt_path).mkdir(parents=True, exist_ok=True)
    Path(fig_path).mkdir(parents=True, exist_ok=True)

    # model construction

    if model_type == 'lstm':
        n_hidden = 64
        n_layers = 1
        model = LSTM(feature_dim, n_hidden, n_categories, n_layers)
    elif model_type == 'rnn':
        n_hidden = 64
        n_layers = 1
        model = RNN(feature_dim, n_hidden, n_categories)
    elif model_type == 'transformer':
        model = Transformer(output_size=n_categories, d_model=feature_dim, nhead=4, d_hid=64,
                     nlayers = 2, dropout = 0.5)
    
    return model, ckpt_path, fig_path


def run_train(model, model_type, ckpt_path, train_data, test_data, max_iter = 1e5):

    # training configs
    lr = 0.001
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    n_iterations = max_iter
    print_every = 100
    save_model_every = 500
    plot_every = 500
    save_record_every = 500

    # Keep track of losses for plotting

    current_loss = 0
    current_val_loss = 0
    all_losses = []
    all_val_losses = []
    all_val_acc = []
    current_correct = 0

    start = time.time()

    start_iteration = 1
    iteration = 1

    for check_i in np.arange(start_iteration - start_iteration % save_model_every, n_iterations+1, save_model_every):
        if check_i == 0 or os.path.exists(f'{ckpt_path}_ckpt_iteration_{int(check_i)}.pt'):
            start_iteration = int(check_i + 1)
            iteration = start_iteration
        else:
            break
            
    while iteration < n_iterations + 1:
        category, line, category_tensor, line_tensor = randomTrainingExample(train_data, test_data)
        
        if iteration != 1 and iteration == start_iteration:    
            prev_model_iteration = int(np.floor(iteration / save_model_every) * save_model_every)
            iteration = prev_model_iteration + 1        
            checkpoint = torch.load(f'{ckpt_path}_ckpt_iteration_{prev_model_iteration}.pt')
            print('restart from checkpoint ', prev_model_iteration, f'{ckpt_path}_ckpt_iteration_{prev_model_iteration}.pt')

            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
            prev_print_iteration = int(np.floor(iteration / save_record_every) * save_record_every)
            if os.path.exists(f'{ckpt_path}_train_loss_iteration_{prev_print_iteration}.npy'):
                all_losses = list(np.load(f'{ckpt_path}_train_loss_iteration_{prev_print_iteration}.npy',allow_pickle=True))
                all_val_losses = list(np.load(f'{ckpt_path}_test_loss_iteration_{prev_print_iteration}.npy',allow_pickle=True))
                all_val_acc = list(np.load(f'{ckpt_path}_test_acc_iteration_{prev_print_iteration}.npy',allow_pickle=True))

        model, optimizer, output, loss = train(model, optimizer, category_tensor, line_tensor, model_type=model_type)
        current_loss += loss.item()
        if iteration % save_model_every == 0:
            torch.save({
                'iteration': iteration,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                }, f'{ckpt_path}_ckpt_iteration_{iteration}.pt')
        
        category, line, category_tensor, line_tensor = randomTrainingExample(train_data, test_data, test=True)
        model, optimizer, output, val_loss = train(model, optimizer, category_tensor, line_tensor, update=False, model_type=model_type)
        current_val_loss += val_loss.item()

        guess, guess_i = categoryFromOutput(output)
        
        correct = '✓' if guess == category else '✗ (%s)' % category
        correct_increment = 1 if guess == category else 0
        current_correct += correct_increment

        # Print iteration number, loss, name and guess
        if iteration % print_every == 0:
            val_acc = current_correct / print_every
            current_correct = 0
            print('%d %d%% (%s) %.4f / %.4f / %.4f : %s %s' % (iteration, iteration / n_iterations * 100, timeSince(start), loss, val_loss, val_acc, guess, correct))

        # Add current loss avg to list of losses
        if iteration % save_record_every == 0:
            all_losses.append(current_loss / save_record_every)
            all_val_losses.append(current_val_loss / save_record_every)
            current_loss = 0
            current_val_loss = 0
            all_val_acc.append(val_acc)
            np.save(f'{ckpt_path}_train_loss_iteration_{iteration}.npy', np.array(all_losses))
            np.save(f'{ckpt_path}_test_loss_iteration_{iteration}.npy', np.array(all_val_losses))
            np.save(f'{ckpt_path}_test_acc_iteration_{iteration}.npy', np.array(all_val_acc))
    
        if iteration % plot_every == 0:
            plt.figure()
            plt.plot(save_record_every * np.arange(len(all_losses)),all_losses,label='train_loss')
            plt.plot(save_record_every * np.arange(len(all_val_losses)),all_val_losses,label='test_loss')
            plt.legend()
            plt.show()
    
        iteration += 1
    
    return model
        
def run_test(model, model_type, train_data, test_data, ckpt_path, n_test_sample = 1000, selected_ckpt = 10000):

    # Keep track of correct guesses in a confusion matrix
    confusion = torch.zeros(n_categories, n_categories)
    correct = 0

    checkpoint = torch.load(f'{ckpt_path}_ckpt_iteration_{selected_ckpt}.pt')
    model.load_state_dict(checkpoint['model_state_dict'])

    # Go through a bunch of examples and record which are correctly guessed
    for i in range(n_test_sample):
        print(i, end='\r')
        category, line, category_tensor, line_tensor = randomTrainingExample(train_data, test_data, test=True)
        output = evaluate(model, line_tensor, model_type)
        guess, guess_i = categoryFromOutput(output)
        category_i = all_categories.index(category)
        confusion[category_i][guess_i] += 1
        if guess == category: 
            correct += 1

    print('',end='\n')
    # Normalize by dividing every row by its sum
    for i in range(n_categories):
        confusion[i] = confusion[i] / confusion[i].sum()
        
    accuracy = correct / n_test_sample
    confusion_accuracy = confusion.diag().mean()
    return accuracy, confusion_accuracy, confusion


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.randn(1, self.hidden_size)
    
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, lstm_layers=1):
        super(LSTM, self).__init__()

        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lstm_layers = lstm_layers
        
        self.lstm = nn.LSTM(input_size,hidden_size,lstm_layers)
        self.lstm2o = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output, hidden = self.lstm(input, hidden)
        output = self.lstm2o(output)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return (torch.randn(self.lstm_layers, self.hidden_size), torch.randn(self.lstm_layers, self.hidden_size))

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class Transformer(nn.Module):

    def __init__(self, output_size: int, d_model: int = 64, nhead: int = 8, d_hid: int = 64,
                 nlayers: int = 2, dropout: float = 0.5):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask):
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        output = self.softmax(output.mean(dim=0))

        return output

