import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
import gensim
import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import time

def preprocess(task):
    # データを取得
    with open('data/squad1.1/train-v1.1.json', encoding='utf-8') as f:
        data_dict = json.load(f)
    # word2vecのモデルを取得
    if task == 'undebiased':
        w2v_model = gensim.models.KeyedVectors.load_word2vec_format(
            './data/GoogleNews-vectors-negative300.bin', binary=True
        )
    elif task == 'debiased':
        w2v_model = gensim.models.KeyedVectors.load_word2vec_format(
            './data/GoogleNews-vectors-negative300-hard-debiased.bin', binary=True
        )
    # context, questionに関する数字を調査
    def isSign(word):
        return len(word) == 1 and not word in w2v_model.key_to_index
    num_context = 0
    num_question = 0
    max_words_context = 0
    max_words_question = 0
    for title in data_dict['data']:
        for paragraph in title['paragraphs']:
            num_context += 1
            num_words_context = len([
                word for word in word_tokenize(paragraph['context'].lower())
                if not isSign(word)
            ])
            if num_words_context > max_words_context:
                max_words_context = num_words_context
            for qa in paragraph['qas']:
                num_question += 1
                num_words_question = len([
                    word for word in word_tokenize(qa['question'].lower())
                    if not isSign(word)
                ])
                if num_words_question > max_words_question:
                    max_words_question = num_words_question
    # X,yを作成
    contexts = torch.zeros(num_context, max_words_context, 300)
    context_idxs = []
    questions = torch.zeros(num_question, max_words_question, 300)
    starts = []
    ends = []
    context_idx = 0
    question_idx = 0
    for title in data_dict['data']:
        for paragraph in title['paragraphs']:
            context = [
                word for word in word_tokenize(paragraph['context'].lower())
                if not isSign(word)
            ]
            for i, word in enumerate(context):
                if word in w2v_model.key_to_index:
                    contexts[context_idx, i] += w2v_model[word]
            for qa in paragraph['qas']:
                question = [
                    word for word in word_tokenize(qa['question'].lower())
                    if not isSign(word)
                ]
                for i, word in enumerate(question):
                    if word in w2v_model.key_to_index:
                        questions[question_idx, i] += w2v_model[word]
                question_idx += 1
                answer = qa['answers'][0]
                context = paragraph['context']
                context = context[:answer['answer_start']] + ' _start_point_ ' + context[answer['answer_start']:]
                context = [
                    word for word in word_tokenize(context.lower())
                    if not isSign(word)
                ]
                start_idx = context.index('_start_point_')
                end_idx = start_idx + len(word_tokenize(answer['text'])) - 1
                starts.append(start_idx)
                ends.append(end_idx)
                context_idxs.append(context_idx)
            context_idx += 1
    return contexts, questions, starts, ends, context_idxs

class MyDataset(Dataset):
    def __init__(self, contexts, context_idxs, questions, starts, ends):
        self.contexts = contexts
        self.context_idxs = context_idxs
        self.questions = questions
        self.starts = starts
        self.ends = ends
        self.num_data = len(starts)

    def __len__(self):
        return self.num_data

    def __getitem__(self, i):
        return self.contexts[self.context_idxs[i]], self.questions[i], self.starts[i], self.ends[i]

class DocReader(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layer, dropout):
        super(DocReader, self).__init__()
        self.P_Enc = nn.LSTM(
            input_dim, hidden_dim, num_layers=num_layer,
            dropout=dropout, bidirectional=True,
            batch_first=True
        )
        self.Q_Enc = nn.LSTM(
            input_dim, hidden_dim, num_layers=num_layer,
            dropout=dropout, bidirectional=True,
            batch_first=True
        )
        self.Q_AccumWeight = nn.Linear(hidden_dim*2, 1, bias=False)
        self.Start_Pred = nn.Linear(hidden_dim*2, hidden_dim*2, bias=False)
        self.End_Pred = nn.Linear(hidden_dim*2, hidden_dim*2, bias=False)

    def forward(self, p, q):
        p, _ = self.P_Enc(p)
        q, _ = self.Q_Enc(q)
        b = F.softmax(self.Q_AccumWeight(q), dim=1)
        q = (q*b).sum(dim=1)
        pred_start = torch.einsum('bnm,bm->bn', self.Start_Pred(p), q)
        pred_end = torch.einsum('bnm,bm->bn', self.End_Pred(p), q)
        return pred_start, pred_end

def eval_exact_match(pred_start, pred_end, y_start, y_end):
    pred_start = pred_start.max(dim=1).indices
    pred_end = pred_end.max(dim=1).indices
    match = torch.logical_and(pred_start==y_start, pred_end==y_end)
    return match.sum() / len(match)

def eval_f1(pred_start, pred_end, y_start, y_end):
    pred_start = pred_start.max(dim=1).indices
    pred_end = pred_end.max(dim=1).indices
    start_diff = pred_start - y_start
    start_diff = torch.where(start_diff > 0, start_diff, 0)
    end_diff = y_end - pred_end
    end_diff = torch.where(end_diff > 0, end_diff, 0)
    TP = (y_end - y_start + 1 - start_diff - end_diff).double()
    TP = torch.where(TP > 0, TP, 1e-10) # TP=0だとf1がnanになる
    recall = TP / (y_end - y_start + 1)
    pred_diff = pred_end - pred_start
    pred_diff = torch.where(pred_diff > 0, pred_diff, 0)
    precision = TP / (pred_diff + 1)
    f1 = 2*recall*precision / (recall + precision)
    return f1.mean()

def train():
    train_loss_log = []
    eval_loss_log = []
    eval_em_log = []
    eval_f1_log = []
    for epoch in range(num_epoch):
        time_start = time.time()
        model.train()
        train_loss = 0
        for X_context, X_question, y_start, y_end in train_loader:
            X_context, X_question = X_context.to(device), X_question.to(device)
            y_start, y_end = y_start.to(device), y_end.to(device)
            optimizer.zero_grad()
            pred_start, pred_end = model(X_context, X_question)
            loss = loss_fn(pred_start, y_start) + loss_fn(pred_end, y_end)
            loss.backward()
            optimizer.step()
            train_loss += loss
        train_loss /= len(train_loader)
        train_loss_log.append(train_loss.cpu().item())

        model.eval()
        eval_loss = 0
        em_total = 0
        f1_total = 0
        with torch.no_grad():
            for X_context, X_question, y_start, y_end in test_loader:
                X_context, X_question = X_context.to(device), X_question.to(device)
                y_start, y_end = y_start.to(device), y_end.to(device)
                pred_start, pred_end = model(X_context, X_question)
                loss = loss_fn(pred_start, y_start) + loss_fn(pred_end, y_end)
                eval_loss += loss
                em = eval_exact_match(pred_start, pred_end, y_start, y_end)
                em_total += em
                f1 = eval_f1(pred_start, pred_end, y_start, y_end)
                f1_total += f1
            eval_loss /= len(test_loader)
            eval_loss_log.append(eval_loss.cpu().item())
            em_total /= len(test_loader)
            eval_em_log.append(em_total.cpu().item())
            f1_total /= len(test_loader)
            eval_f1_log.append(f1_total.cpu().item())
            print('Epoch{}({:.2f}sec)'.format(epoch+1, time.time()-time_start))
        if isModeling:
            print('loss train:{:.4f}, test:{:.4f}'.format(train_loss, eval_loss))
            print('score EM:{:.4f}, F1:{:.4f}'.format(em_total, f1_total))
    
    best_em = max(eval_em_log)
    best_f1 = max(eval_f1_log)
    if not isModeling:
        print('Experiment', seed+1)
    print('best EM:{:.4f}(Epoch{}), F1:{:.4f}(Epoch{})'.format(
        best_em, eval_em_log.index(best_em)+1,
        best_f1, eval_f1_log.index(best_f1)+1
    ))
    if isModeling:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(range(1, num_epoch+1), train_loss_log, label='train loss')
        ax.plot(range(1, num_epoch+1), eval_loss_log, label='eval loss')
        fig.suptitle('loss log')
        fig.legend()
        fig.savefig('figs/QA_{}_loss.png'.format(task))
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(range(1, num_epoch+1), eval_em_log, label='Exact Match')
        ax.plot(range(1, num_epoch+1), eval_f1_log, label='F1')
        fig.suptitle('acc log')
        fig.legend()
        fig.savefig('figs/QA_{}_acc.png'.format(task))
    if not isModeling:
        df = pd.read_csv('outputs/QA_results_{}.csv'.format(task))
        df.loc[str(seed)] = [best_em, best_f1]
        df.to_csv('outputs/QA_results_{}.csv'.format(task), index=False)

if __name__ == '__main__':
    if not os.path.isdir('.data'):
        os.mkdir('.data')
    if not os.path.isdir('figs'):
        os.mkdir('figs')
    if not os.path.isdir('outputs'):
        os.mkdir('outputs')

    # パラメータ
    task = 'debiased'
    isModeling = False
    saved = False

    num_experiment = 10
    if isModeling:
        num_experiment = 1

    num_epoch = 20
    lr = 0.01
    train_batch_size = 292
    test_batch_size = 292
    num_workers = 2
    pin_memory = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    print("device:", device)

    if not isModeling and not os.path.isfile('outputs/QA_results_{}.csv'.format(task)):
        df = pd.DataFrame(columns=['{}_em'.format(task), '{}_f1'.format(task)])
        df.to_csv('outputs/QA_results_{}.csv'.format(task), index=False)

    print(task)
    if isModeling:
        print('preprocessing')
    if not saved:
        contexts, questions, starts, ends, context_idxs = preprocess(task)
        torch.save(contexts, 'outputs/QA_context_emb_{}.pt'.format(task))
        torch.save(questions, 'outputs/QA_question_emb_{}.pt'.format(task))
        df = pd.DataFrame({'starts':starts, 'ends':ends, 'context_idxs':context_idxs})
        df.to_csv('outputs/QA_label_index_{}.csv'.format(task), index=False)
    else:
        contexts = torch.load('outputs/QA_context_emb_{}.pt'.format(task))
        questions = torch.load('outputs/QA_question_emb_{}.pt'.format(task))
        df = pd.read_csv('outputs/QA_label_index_{}.csv'.format(task))
        starts = list(df['starts'])
        ends = list(df['ends'])
        context_idxs = list(df['context_idxs'])
    
    for seed in range(num_experiment):
        torch.manual_seed(seed)
        np.random.seed(seed)
        train_context_idxs, test_context_idxs, train_questions, test_questions, train_starts, test_starts, train_ends, test_ends = train_test_split(
            context_idxs, questions, starts, ends, test_size=0.2, random_state=seed
        )
        train_set = MyDataset(contexts, train_context_idxs, train_questions, train_starts, train_ends)
        test_set = MyDataset(contexts, test_context_idxs, test_questions, test_starts, test_ends)
        train_loader = DataLoader(
            train_set,
            batch_size=train_batch_size, shuffle=True, drop_last=True,
            num_workers=num_workers, pin_memory=pin_memory
        )
        test_loader = DataLoader(
            test_set,
            batch_size=test_batch_size,
            num_workers=num_workers, pin_memory=pin_memory
        )
        if isModeling:
            print('training')
        model = DocReader(300, 128, 3, 0.3).to(device)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adamax(model.parameters(), lr=lr)
        train()