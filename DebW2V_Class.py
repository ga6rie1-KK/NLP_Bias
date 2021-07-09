import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchtext import data
from torchtext.datasets import AG_NEWS, DBpedia, YelpReviewPolarity
import os
import gensim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from sklearn.metrics import confusion_matrix

os.environ['KMP_DUPLICATE_LIB_OK']='True'

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print("device:", device)

# PyTorchのデータセットはデフォルトでは.dataに保存される
if not os.path.isdir('.data'):
    os.mkdir('.data')
if not os.path.isdir('figs'):
    os.mkdir('figs')

print('preparing data')
w2v_undebiased = gensim.models.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True)
w2v_debiased = gensim.models.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300-hard-debiased.bin', binary=True)
w2v_dim = 300

def trans_text2vec(text, w2v_dim, w2v_model):
    x = torch.zeros(w2v_dim)
    count = 0
    for i, word in enumerate(text.split()):
        if word[-1] in [',', '.']:
            word = word[:-1]
        if word in w2v_model.key_to_index:
            x += w2v_model[word]
            count += 1
    if count > 0:
        x /= count
    return x

##### defining dataset #####
# dataset_name = 'AGNEWS'
# train_set, test_set = AG_NEWS()
# num_class = 4 # 1~4
dataset_name = 'DBpedia'
train_set, test_set = DBpedia()
num_class = 14 # 1~14
# dataset_name = 'YelpReviewPolarity'
# train_set, test_set = YelpReviewPolarity()
# num_class = 2 # 1,2
############################

##### parameters #####
batch_size = 1000
num_epoch = 100
num_experiment = 100
isModeling = False
######################

X_udb_train = torch.zeros(len(train_set), w2v_dim)
X_udb_test = torch.zeros(len(test_set), w2v_dim)

X_db_train = torch.zeros(len(train_set), w2v_dim)
X_db_test = torch.zeros(len(test_set), w2v_dim)

y_train = torch.zeros(len(train_set), dtype=int)
y_test = torch.zeros(len(test_set), dtype=int)

text_list = []

for i, (label, text) in enumerate(train_set):
    X_udb_train[i] += trans_text2vec(text, w2v_dim, w2v_undebiased)
    X_db_train[i] += trans_text2vec(text, w2v_dim, w2v_debiased)
    y_train[i] += label - 1

for i, (label, text) in enumerate(test_set):
    X_udb_test[i] += trans_text2vec(text, w2v_dim, w2v_undebiased)
    X_db_test[i] += trans_text2vec(text, w2v_dim, w2v_debiased)
    y_test[i] += label - 1
    text_list.append(text)

class MyDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.num_data = len(X)

    def __len__(self):
        return self.num_data

    def __getitem__(self, i):
        x = self.X[i]
        y = self.y[i]
        return x, y

udb_train_set = MyDataset(X_udb_train, y_train)
udb_test_set = MyDataset(X_udb_test, y_test)
db_train_set = MyDataset(X_db_train, y_train)
db_test_set = MyDataset(X_db_test, y_test)

udb_train_loader = DataLoader(udb_train_set, batch_size=batch_size, shuffle=True)
udb_test_loader = DataLoader(udb_test_set, batch_size=len(test_set))
db_train_loader = DataLoader(db_train_set, batch_size=batch_size, shuffle=True)
db_test_loader = DataLoader(db_test_set, batch_size=len(test_set))

class Classifier(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, num_class):
        super(Classifier, self).__init__()
        self.ln1 = nn.Linear(input_dim, hidden_dim1)
        self.ln2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.ln3 = nn.Linear(hidden_dim2, num_class)
    
    def forward(self, x):
        x = self.ln1(x)
        x = F.relu(x)
        x = self.ln2(x)
        x = F.relu(x)
        x = self.ln3(x)
        return x

print('\nundebiased model')
if not isModeling:
    udb_result = []
    udb_preds = []
for e in range(num_experiment):
    torch.manual_seed(e)

    model = Classifier(w2v_dim, 150, 50, num_class).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

    train_loss_log = []
    eval_loss_log = []
    eval_acc_log = []
    best_acc = 0
    for epoch in range(num_epoch):
        if isModeling:
            print('Epoch {}'.format(epoch+1))
        model.train()
        train_loss = 0
        for X, y in udb_train_loader:
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss
        train_loss /= len(udb_train_loader)
        train_loss_log.append(train_loss.detach().cpu().item())
        if isModeling:
            print('train loss:{:.4}'.format(train_loss))

        model.eval()
        with torch.no_grad():
            for X, y in udb_test_loader:
                X = X.to(device)
                y = y.to(device) 
                pred = model(X)
                loss = loss_fn(pred, y)
                acc = (torch.max(pred, dim=1).indices == y).sum() / len(udb_test_set)
        eval_loss_log.append(loss.detach().cpu().item())
        eval_acc_log.append(acc.detach().cpu().item())
        if isModeling:
            print('test loss:{:.4}, acc:{:.4}'.format(loss, acc))
        if acc > best_acc:
            best_acc = acc
            best_pred = pred

    print('Experiment{} max acc: Epoch {}, {:.4}'.format(e+1, eval_acc_log.index(best_acc)+1, best_acc))
    if not isModeling:
        udb_result.append(best_acc.cpu().item())
        udb_preds.append(list(best_pred.max(dim=1).indices.cpu().numpy()))
    if isModeling:
        break

if isModeling:
    print('ploting')
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(range(1, 101), train_loss_log, label='train loss')
    ax.plot(range(1, 101), eval_loss_log, label='eval loss')
    fig.suptitle('undebiased model loss log')
    fig.legend()
    fig.savefig('figs/udb_loss.png')

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(range(1, 101), eval_acc_log)
    fig.suptitle('undebiased model acc log')
    fig.savefig('figs/udb_acc.png')

print('\ndebiased model')
if not isModeling:
    db_result = []
    db_preds = []
for e in range(num_experiment):
    torch.manual_seed(e)

    model = Classifier(w2v_dim, 150, 50, num_class).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

    train_loss_log = []
    eval_loss_log = []
    eval_acc_log = []
    best_acc = 0
    for epoch in range(num_epoch):
        if isModeling:
            print('Epoch {}'.format(epoch+1))
        model.train()
        train_loss = 0
        for X, y in db_train_loader:
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss
        train_loss /= len(db_train_loader)
        train_loss_log.append(train_loss.detach().cpu().item())
        if isModeling:
            print('train loss:{:.4}'.format(train_loss))

        model.eval()
        with torch.no_grad():
            for X, y in db_test_loader:
                X = X.to(device)
                y = y.to(device) 
                pred = model(X)
                loss = loss_fn(pred, y)
                acc = (torch.max(pred, dim=1).indices == y).sum() / len(db_test_set)
        eval_loss_log.append(loss.detach().cpu().item())
        eval_acc_log.append(acc.detach().cpu().item())
        if isModeling:
            print('test loss:{:.4}, acc:{:.4}'.format(loss, acc))
        if acc > best_acc:
            best_acc = acc
            best_pred = pred

    print('Experiment{} max acc: Epoch {}, {:.4}'.format(e+1, eval_acc_log.index(best_acc)+1, best_acc))
    if not isModeling:
        db_result.append(best_acc.cpu().item())
        db_preds.append(list(best_pred.max(dim=1).indices.cpu().numpy()))
    if isModeling:
        break


if isModeling:
    print('ploting')
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(range(1, 101), train_loss_log, label='train loss')
    ax.plot(range(1, 101), eval_loss_log, label='eval loss')
    fig.suptitle('debiased model loss log')
    fig.legend()
    fig.savefig('figs/db_loss.png')

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(range(1, 101), eval_acc_log)
    fig.suptitle('debiased model acc log')
    fig.savefig('figs/db_acc.png')

if not isModeling:
    today = datetime.date.today()
    today = today.strftime('%Y%m%d')
    df = pd.DataFrame({'undebiased':udb_result, 'debiased':db_result})
    df.to_csv('outputs/{}_{}_acc.csv'.format(dataset_name, today), index=False)
    # 最後の列は正解ラベル
    udb_preds.append(list(y_test.numpy()))
    df = pd.DataFrame(udb_preds).T
    df.to_csv('outputs/{}_{}_pred_udb.csv'.format(dataset_name, today), index=False)
    db_preds.append(list(y_test.numpy()))
    df = pd.DataFrame(db_preds).T
    df.to_csv('outputs/{}_{}_pred_db.csv'.format(dataset_name, today), index=False)
    with open('outputs/{}_{}_txtLst.txt'.format(dataset_name, today), mode='w', encoding='utf-8') as f:
        f.write('\n\n\n'.join(text_list))