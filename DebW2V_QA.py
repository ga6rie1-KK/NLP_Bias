import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize, sent_tokenize
import gensim
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os

def preprocess(task):
    with open('data/squad1.1/train-v1.1.json', encoding='utf-8') as f:
        data_dict = json.load(f)
    if task == 'undebiased':
        w2v_model = gensim.models.KeyedVectors.load_word2vec_format(
            './data/GoogleNews-vectors-negative300.bin', binary=True
        )
    elif task == 'debiased':
        w2v_model = gensim.models.KeyedVectors.load_word2vec_format(
            './data/GoogleNews-vectors-negative300-hard-debiased.bin', binary=True
        )
    max_num_sentence = 0
    num_context = 0
    num_question = 0
    for title in data_dict['data']:
        for paragraph in title['paragraphs']:
            num_context += 1
            num_sentence = len(sent_tokenize(paragraph['context']))
            if num_sentence > max_num_sentence:
                max_num_sentence = num_sentence
            for qa in paragraph['qas']:
                num_question += 1
    questions = torch.zeros(num_question, 300)
    contexts = torch.zeros(num_context, max_num_sentence, 300)
    context_idxs = []
    answers = []
    context_id = 0
    question_id = 0
    for title in data_dict['data']:
        for paragraph in title['paragraphs']:
            sentences = sent_tokenize(paragraph['context'])
            for i, sentence in enumerate(sentences):
                words = [word for word in word_tokenize(sentence.lower()) if word in w2v_model.key_to_index]
                if len(words) > 0:
                    contexts[context_id, i] += w2v_model[words].mean(axis=0)
                else:
                    contexts[context_id, i] += torch.zeros(300)
            for qa in paragraph['qas']:
                questions[question_id] += w2v_model[[word for word in word_tokenize(qa['question'].lower()) if word in w2v_model.key_to_index]].mean(axis=0)
                question_id += 1
                answer = qa['answers'][0]
                context = paragraph['context']
                context = context[:answer['answer_start']] + '_answer_tag_' + context[answer['answer_start']:]
                sentences = sent_tokenize(context)
                for sentence in sentences:
                    if '_answer_tag_' in sentence:
                        answers.append(sentences.index(sentence))
                context_idxs.append(context_id)
            context_id += 1
    return questions, contexts, context_idxs, answers, max_num_sentence

class MyDataset(Dataset):
    def __init__(self, questions, contexts, context_idxs, answers):
        self.questions = questions
        self.contexts = contexts
        self.context_idxs = context_idxs
        self.answers = answers
        self.num_data = len(answers)

    def __len__(self):
        return self.num_data

    def __getitem__(self, i):
        question = self.questions[i]
        context = self.contexts[self.context_idxs[i]]
        answer = self.answers[i]
        return question, context, answer

# パターン1
class QAModel1(nn.Module):
    def __init__(self, w2v_dim, hid1, hid2, num_sentence, hid3):
        super(QAModel1, self).__init__()
        self.QEnc1 = nn.Linear(w2v_dim, hid1)
        self.QEnc2 = nn.Linear(hid1, hid2)
        self.SEnc1 = nn.Linear(w2v_dim, hid1)
        self.SEnc2 = nn.Linear(hid1, hid2)
        self.Classifier1 = nn.Linear(hid2*(1+num_sentence), hid3)
        self.Classifier2 = nn.Linear(hid3, num_sentence)
    
    def forward(self, q, s, batch_size):
        q = self.QEnc1(q)
        q = F.relu(q)
        q = self.QEnc2(q)
        s = self.SEnc1(s)
        s = F.relu(s)
        s = self.SEnc2(s)
        x = torch.cat((q, s.view(batch_size, -1)), dim=1)
        x = self.Classifier1(x)
        x = F.relu(x)
        x = self.Classifier2(x)
        return x

# パターン2
class QAModel2(nn.Module):
    def __init__(self, w2v_dim, hid1, hid2, hid3, num_sentence):
        super(QAModel2, self).__init__()
        self.ln1 = nn.Linear(w2v_dim*2, hid1)
        self.ln2 = nn.Linear(hid1, hid2)
        self.ln3 = nn.Linear(hid2, hid3)
        self.ln4 = nn.Linear(hid3, 1)
        self.num_sentence = num_sentence
    
    def forward(self, q, s, batch_size):
        q = torch.stack([q for _ in range(self.num_sentence)], dim=1)
        x = torch.cat((q, s), dim=2)
        x = self.ln1(x)
        x = F.relu(x)
        x = self.ln2(x)
        x = F.relu(x)
        x = self.ln3(x)
        x = F.relu(x)
        x = self.ln4(x)
        x = x.view(batch_size, -1)
        return x


def train():
    if not isModeling:
        print('Seed', seed)
    train_loss_log = []
    eval_loss_log = []
    eval_acc_log = []
    for epoch in range(num_epoch):
        time_start = time.time()
        model.train()
        train_loss = 0
        for X_question, X_context, y in train_loader:
            X_question, X_context, y = X_question.to(device), X_context.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(X_question, X_context, train_batch_size)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss
        train_loss /= len(train_loader)
        train_loss_log.append(train_loss.cpu().item())

        model.eval()
        eval_loss = 0
        acc_total = 0
        with torch.no_grad():
            for X_question, X_context, y in test_loader:
                X_question, X_context, y = X_question.to(device), X_context.to(device), y.to(device)
                pred = model(X_question, X_context, test_batch_size)
                loss = loss_fn(pred, y)
                eval_loss += loss
                acc = (torch.max(pred, dim=1).indices == y).sum() / test_batch_size
                acc_total += acc
            eval_loss /= len(test_loader)
            eval_loss_log.append(eval_loss.cpu().item())
            acc_total /= len(test_loader)
            eval_acc_log.append(acc_total.cpu().item())
        if isModeling:
            print(
                'Epoch{}'.format(epoch+1)
                + ' '*(4 - len(str(epoch+1)))
                + 'train loss:{:.4f}, test loss:{:.4f}, acc:{:.4f} ({:.2f}sec)'.format(
                    train_loss, eval_loss, acc_total, time.time()-time_start
                )
            )
    best_acc = max(eval_acc_log)
    print('best acc:{:.4f}(Epoch{})'.format(best_acc, eval_acc_log.index(best_acc)+1))
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
        ax.plot(range(1, num_epoch+1), eval_acc_log, label='acc')
        fig.suptitle('acc log')
        fig.legend()
        fig.savefig('figs/QA_{}_acc.png'.format(task))
    if not isModeling:
        df = pd.read_csv('outputs/QA_results_{}.csv'.format(task))
        df.loc[seed] = [best_acc, eval_acc_log.index(best_acc)+1]
        df.to_csv('outputs/QA_results_{}.csv'.format(task), index=False)


if __name__ == '__main__':
    task = 'undebiased'
    isModeling = False
    saved = True
    pattern = 2

    seeds = range(40, 50)
    if isModeling:
        seeds = range(1)

    if task == 'undebiased':
        num_epoch = 1000
    elif task == 'debiased':
        num_epoch = 2000
    lr = 0.05
    train_batch_size = 10000
    test_batch_size = 17520
    num_workers = 0
    pin_memory = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    print("device:", device)

    if not isModeling and not os.path.isfile('outputs/QA_results_{}.csv'.format(task)):
        df = pd.DataFrame(columns=['acc', 'epoch'])
        df.to_csv('outputs/QA_results_{}.csv'.format(task), index=False)

    # questions(num_qa, w2v_dim):questionの分散表現
    # contexts(num_context, max_num_sentence, w2v_dim):contextの各文ごとの分散表現
    # context_idxs(num_qa):各qaに対してどのcontextが対応するか
    # answers(num_qa):何個目の文がanswerか
    if not saved:
        print('preprocessing')
        questions, contexts, context_idxs, answers, max_num_sentence = preprocess(task)
        torch.save(questions, 'outputs/QA_question_emb_{}.pt'.format(task))
        torch.save(contexts, 'outputs/QA_context_emb_{}.pt'.format(task))
        pd.DataFrame({'answers':answers, 'context_idxs':context_idxs}).to_csv('outputs/QA_label_index_{}.csv'.format(task), index=False)
    else:
        questions = torch.load('outputs/QA_question_emb_{}.pt'.format(task))
        contexts = torch.load('outputs/QA_context_emb_{}.pt'.format(task))
        df = pd.read_csv('outputs/QA_label_index_{}.csv'.format(task))
        answers = list(df['answers'])
        context_idxs = list(df['context_idxs'])
        max_num_sentence = contexts.shape[1]

    print(task)
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        train_questions, test_questions, train_context_idxs, test_context_idxs, train_answers, test_answers = train_test_split(
            questions, context_idxs, answers, test_size=0.2, random_state=seed
        )
        train_set = MyDataset(train_questions, contexts, train_context_idxs, train_answers)
        test_set = MyDataset(test_questions, contexts, test_context_idxs, test_answers)
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
        if pattern == 1:
            model = QAModel1(300, 100, 10, max_num_sentence, 100).to(device)
        elif pattern == 2:
            model = QAModel2(300, 300, 100, 10, max_num_sentence).to(device)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        train()