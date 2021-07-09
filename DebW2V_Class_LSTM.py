import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchtext import data
from torchtext.datasets import AG_NEWS, IMDB
import os
import gensim
import numpy as np
import matplotlib.pyplot as plt

#os.environ['KMP_DUPLICATE_LIB_OK']='True'

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print("device:", device)

# PyTorchのデータセットはデフォルトでは.dataに保存される
if not os.path.isdir('.data'):
    os.mkdir('.data')

train_set, test_set = AG_NEWS()
X_train = []
y_train = []
X_test = []
y_test = []
max_text_size = 0
for (label, text) in train_set:
    X_train.append(text)
    y_train.append(label)
    text_size = len(text.split())
    if  text_size > max_text_size:
        max_text_size = text_size
for (label, text) in test_set:
    X_test.append(text)
    y_test.append(label)
    text_size = len(text.split())
    if text_size > max_text_size:
        max_text_size = text_size
num_class = len(set(y_train + y_test)) #1,2,3,4
w2v_undebiased = gensim.models.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True)
w2v_debiased = gensim.models.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300-hard-debiased.bin', binary=True)
w2v_dim = 300
def trans_text2vec(text, max_text_size, w2v_dim, w2v_model):
    x = torch.zeros(max_text_size, w2v_dim)
    for i, word in enumerate(text.split()):
        if word[-1] in [',', '.']:
            word = word[:-1]
        if word in w2v_model.key_to_index:
            x[i] += w2v_model[word]
    return x

class MyDataset(Dataset):
    def __init__(self, X, y, max_text_size, w2v_dim, w2v_model):
        self.X = X
        self.y = y
        self.num_data = len(X)
        self.max_text_size = max_text_size
        self.w2v_dim = w2v_dim
        self.w2v_model = w2v_model

    def __len__(self):
        return self.num_data

    def __getitem__(self, i):
        x = trans_text2vec(self.X[i],
                           self.max_text_size,
                           self.w2v_dim,
                           self.w2v_model)
        y = self.y[i]
        return x, y

train_set = MyDataset(X_train, y_train, max_text_size, w2v_dim, w2v_undebiased)
test_set = MyDataset(X_test, y_test, max_text_size, w2v_dim, w2v_undebiased)
batch_size = 1000
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=len(test_set))

print('defining model')
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_class):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim)
        self.ln = nn.Linear(hidden_dim, num_class)
    
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.ln(x[:, -1])
        return x

model = LSTMClassifier(w2v_dim, 100, num_class).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

train_loss_log = []
eval_loss_log = []
eval_acc_log = []
for epoch in range(100):
    print('Epoch {}'.format(epoch+1))
    print('training')
    model.train()
    train_loss = 0
    for X, y in train_loader:
        X = X.to(device)
        y = (y-1).to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss
    train_loss /= len(train_loader)
    train_loss_log.append(train_loss)
    print('train loss:{:.4}'.format(train_loss))

    print('evaluating')
    model.eval()
    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device)
            y = (y-1).to(device) 
            pred = model(X)
            loss = loss_fn(pred, y)
            acc = (torch.max(pred, dim=1).indices == y).sum() / len(test_set)
    eval_loss_log.append(loss)
    eval_acc_log.append(acc)
    print('test loss:{:.4}, acc:{:.4}'.format(loss, acc))
    print('')

# plt.plot(range(1, 101), train_loss_log)
# plt.plot(range(1, 101), eval_loss_log)
# plt.show()

# plt.plot(range(1, 101), eval_acc_log)
# plt.show()