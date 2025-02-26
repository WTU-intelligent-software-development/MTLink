from __future__ import print_function
from __future__ import absolute_import
import argparse
import os
import numpy as np
import math
import time
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from torch.utils.data import DataLoader, Sampler, random_split
import matplotlib.pyplot as plt
import torch.nn.init as weight_init
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
import logging
from torch.nn import CrossEntropyLoss
import pickle
import pandas as pd
from pathlib import Path
from typing import List, Tuple

from pandas import DataFrame
from transformers import AutoTokenizer, AutoModel
from transformers import (RobertaConfig, RobertaModel,
                          AdamW, BertConfig, BertModel, BertTokenizer,
                          RobertaTokenizer, get_linear_schedule_with_warmup, BertForMaskedLM,
                          get_cosine_with_hard_restarts_schedule_with_warmup)
from ad_Multi_Model import Multi_Model

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
os.environ["CUDA_LAUNCH_BLOCKING"] = '1'

Device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = RobertaTokenizer.from_pretrained("../Dstill/microsoft/codebert-base")


def get_args():
    parser = argparse.ArgumentParser(description="EALink.py")
    parser.add_argument("--end_epoch", type=int, default=400,
                        help="Epoch to stop training.")

    parser.add_argument("--tra_batch_size", type=int, default=16,
                        help="Batch size set during training")

    parser.add_argument("--val_batch_size", type=int, default=16,
                        help="Batch size set during predicting")

    parser.add_argument("--tes_batch_size", type=int, default=16,
                        help="Batch size set during predicting")
    parser.add_argument("--output_model", type=str, default='',
                        help="The path to save model")
    return parser.parse_args()


opt = get_args()


def text2vec(seqs):
    texttoken_id = []
    max_seq_len = 35
    # print(seqs)
    textoken = []
    for seq in seqs:
        textoken = textoken + [tokenizer.cls_token] + seq
    tokens_ids = tokenizer.convert_tokens_to_ids(textoken)
    texttoken_id = tokens_ids[:max_seq_len]
    texttoken_id.extend([0 for _ in range(max_seq_len - len(texttoken_id))])
    return texttoken_id


def code2vec(codes, isdiff):
    codetoken_id = []
    max_code_len = 300
    max_diff_len = 500
    codes = eval(codes)
    code_tokens = []
    if isdiff:
        for code in codes:
            code_id = tokenizer.convert_tokens_to_ids([tokenizer.cls_token] + code)
            code_id = code_id[:80]
            code_id.extend([0 for _ in range(80 - len(code_id))])
            codetoken_id.extend(code_id)
        codetoken_id = codetoken_id[:max_diff_len]
        codetoken_id.extend([0 for _ in range(max_diff_len - len(codetoken_id))])
    else:
        for code in codes:
            code_tokens = code_tokens + [tokenizer.cls_token] + code[:80]
        tokens_ids = tokenizer.convert_tokens_to_ids(code_tokens)
        codetoken_id = tokens_ids[:max_code_len]
        codetoken_id.extend([0 for _ in range(max_code_len - len(codetoken_id))])
    # print(len(codetoken_id))
    return codetoken_id


def make_batches(df, mode='train'):
    # input_batch, output_batch, target_batch = [],[],[]
    msg_batch1, code_batch1, desc_batch1, tg_batch1, diff_batch1, la_batch1, num_batch1, commitid_batch1, issueid_batch1 = [], [], [], [], [], [], [], [], []
    logging.info("Loaded the file")
    max_len = 20
    for index, row in df.iterrows():
        commit = text2vec(eval(row['message_processed']))  # 得到tokenid
        issue = text2vec(eval(row['summary_processed']) + eval(row['description_processed']))
        if len(eval(row['codelist_processed'])) == 0:
            code = code2vec(row['Diff_processed'], False)
        else:
            code = code2vec(row['codelist_processed'], False)
        tg = float(row['target'])

        issueid = int(row['issue_id'])
        commitid = row['hash']
        msg_batch1.append(commit)
        code_batch1.append(code)
        desc_batch1.append(issue)
        tg_batch1.append(tg)

        issueid_batch1.append(issueid)
        commitid_batch1.append(commitid)
        diff = code2vec(row['Diff_processed'], True)
        label = eval(row['labelist'])
        label = label[:max_len]
        label.extend([3 for _ in range(max_len - len(label))])
        num = int(row['num'])
        diff_batch1.append(diff)
        la_batch1.append(label)
        num_batch1.append(num)

    print(len(code_batch1[0]))
    print(len(tg_batch1))

    return torch.LongTensor(msg_batch1), torch.LongTensor(code_batch1), torch.LongTensor(desc_batch1), torch.LongTensor(
        tg_batch1), torch.LongTensor(diff_batch1), torch.LongTensor(la_batch1), torch.LongTensor(
        num_batch1), torch.LongTensor(issueid_batch1), commitid_batch1


class MyDataSet(Data.Dataset):
    def __init__(self, message_inputs, code_inputs, desc_inputs, target, diff, label, num, issueid, commitid):
        super(MyDataSet, self).__init__()
        self.message_inputs = message_inputs
        self.code_inputs = code_inputs
        self.desc_inputs = desc_inputs
        self.target = target
        self.diff = diff
        self.label = label
        self.num = num
        self.issueid = issueid
        self.commitid = commitid

    def __len__(self):
        return self.message_inputs.shape[0]

    def __getitem__(self, idx):
        return self.message_inputs[idx], self.code_inputs[idx], self.desc_inputs[idx], self.target[idx], self.diff[idx], \
               self.label[idx], self.num[idx], self.issueid[idx], self.commitid[idx]


class MySampler(Sampler):
    def __init__(self, dataset, batchsize):
        super(Sampler, self).__init__()
        self.dataset = dataset
        self.batch_size = batchsize  # batchsize
        self.indices = range(len(dataset))
        self.count = int(len(dataset) / self.batch_size)  # number of batch

    def __iter__(self):
        for i in range(self.count):
            yield self.indices[i * self.batch_size: (i + 1) * self.batch_size]

    def __len__(self):
        return self.count


def lsplit(list1, n):
    output = [list1[i:i + 80] for i in range(0, 300, 80)]
    return output[:n]


def lremove(list_i, value):
    j = 0
    for i in range(len(list_i)):
        if list_i[j] == value:
            list_i.pop(j)
        else:
            j += 1
    return list_i


def allindex(data, n):
    index = [i for i, x in enumerate(data) if x == n]
    return index


def allindex_cl(data, n):
    list1 = [0] * 16
    index = [i for i, x in enumerate(data) if x == n]
    kk = random.choice(index)
    list1[kk] = 1
    return list1


def results_to_df(res: List[Tuple]) -> DataFrame:
    df = pd.DataFrame()
    df['s_id'] = [x[0] for x in res]
    df['t_id'] = [x[1] for x in res]
    df['pred'] = [x[2] for x in res]
    df['label'] = [x[3] for x in res]
    group_sort = df.groupby(["s_id"]).apply(
        lambda x: x.sort_values(["pred"], ascending=False)).reset_index(drop=True)
    return group_sort


def MRR(data_frame):
    group_tops = data_frame.groupby('s_id')
    mrr_sum = 0
    for s_id, group in group_tops:
        rank = 0
        for i, (index, row) in enumerate(group.iterrows()):
            rank += 1
            if row['label'] == 1:
                mrr_sum += 1.0 / rank
                break
    return mrr_sum / len(group_tops)


if __name__ == '__main__':

    trans_model = Multi_Model()
    trans_model = torch.nn.DataParallel(trans_model).cuda()
    # adam
    trans_optimizer = optim.Adam(trans_model.parameters(), lr=4e-05, eps=1e-08)
    # 读取训练集和验证集
    df = pd.read_csv('../data/new_data/netbeans_link.csv')
    train_df_sum = df.loc[df['train_flag'] == 1]
    cnt = len(train_df_sum) / 4
    train_df = train_df_sum.loc[:cnt * 3 - 1]
    valid_df = train_df_sum.loc[cnt * 3:]
    message_input1, code_input1, desc_input1, target1, diff1, label1, nums1, issueid1, commitid1 = make_batches(
        df=train_df)
    message_input2, code_input2, desc_input2, target2, diff2, label2, nums2, issueid2, commitid2 = make_batches(
        df=valid_df)

    logging.info("Loaded the file done")
    # data processing
    train_data = MyDataSet(message_input1, code_input1, desc_input1, target1, diff1, label1, nums1, issueid1, commitid1)
    valid_data = MyDataSet(message_input2, code_input2, desc_input2, target2, diff2, label2, nums2, issueid2, commitid2)

    my_sampler1 = MySampler(train_data, opt.tra_batch_size)
    my_sampler2 = MySampler(valid_data, opt.val_batch_size)
    train_data_loader = Data.DataLoader(train_data, batch_sampler=my_sampler1)
    valid_data_loader = Data.DataLoader(valid_data, batch_sampler=my_sampler2)
    best_test_loss = float("inf")

    train_ls, valid_ls = [], []

    avg_cost = np.zeros([opt.end_epoch, len(label1)], dtype=np.float32)
    lambda_weight = np.ones([3, opt.end_epoch])
    T = 2
    for epoch in range(opt.end_epoch):
        cost = np.zeros(2, dtype=np.float32)
        if epoch == 0 or epoch == 1 or epoch == 2:
            pass
        else:
            sum_w = []
            w_1 = avg_cost[epoch - 1, 0] / avg_cost[epoch - 2, 0]
            w_2 = avg_cost[epoch - 1, 1] / avg_cost[epoch - 2, 1]

            lambda_weight[0, epoch] = 2*np.exp(w_1 / T) / (np.exp(w_1 / T) + np.exp(w_2 / T))
            lambda_weight[1, epoch] = 2*np.exp(w_2 / T) / (np.exp(w_1 / T) + np.exp(w_2 / T))

        epoch_loss = 0
        t3 = time.time()
        trans_model.train()  # 训练模型
        match_tra_order = 0
        for message_inputs, code_inputs, desc_inputs, target, diff_inputs, labels, nums, issueid, commitid in train_data_loader:
            difflist = diff_inputs.tolist()
            numlist = nums.tolist()
            desclist = desc_inputs.tolist()
            labelist = labels.tolist()
            issuetest = []
            diff = []
            label = []

            idlist = issueid.tolist()
            p = []
            for i in range(0, len(difflist)):
                diff_split = lsplit(difflist[i], numlist[i])
                diff.extend(diff_split)
                len1 = len(diff_split)
                label.extend(lremove(labelist[i], 3)[:len1])
                for numx in range(0, len1):
                    issuetest.append(desclist[i])
            idlist = issueid.tolist()
            for j in range(0, len(idlist)):
                if target[j]:
                    p.append(allindex_cl(issueid, idlist[j]))
            issuetest = torch.LongTensor(issuetest)
            # 创建张量
            diff = torch.LongTensor(diff)
            label = torch.LongTensor(label)
            p = torch.LongTensor(p)
            message_inputs, code_inputs, desc_inputs, target, issuetest, diff, label, p = message_inputs.cuda(), code_inputs.cuda(), desc_inputs.cuda(), target.cuda(), issuetest.cuda(), diff.cuda(), label.cuda(), p.cuda()
            loss, c_l, n_l, sim_score = trans_model(message_inputs, code_inputs, desc_inputs,
                                                    target.float(), issuetest, diff,
                                                    label, p)

            cost[0] = loss[0].item()
            cost[1] = loss[1].item()
            avg_cost[epoch, :2] += cost[:2] / len(train_data_loader)

            loss = sum([lambda_weight[i, epoch] * loss[i] for i in range(3)])

            trans_optimizer.zero_grad()  # 清楚梯度
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=trans_model.parameters(), max_norm=10, norm_type=2)  # 要不要删除
            trans_optimizer.step()  # 反向传播
            epoch_loss += loss.item()
        train_avg_loss = epoch_loss / len(train_data_loader)
        print(epoch)
        print(lambda_weight[0, epoch], lambda_weight[1, epoch], lambda_weight[2, epoch])
        print('\ttrain loss: ', '{:.4f}'.format(train_avg_loss))
        train_ls.append(train_avg_loss)
        torch.cuda.synchronize()
        t4 = time.time()
        print("At the train epoch, cost time:%f" % (t4 - t3))

        # eval
        epoch_loss = 0
        trans_model.eval()

        match_val_order = 0
        res = []
        mrr = 0
        with torch.no_grad():
            for message_inputs, code_inputs, desc_inputs, target, diff_inputs, labels, nums, issueid, commitid in valid_data_loader:
                difflist = diff_inputs.tolist()
                numlist = nums.tolist()
                desclist = desc_inputs.tolist()
                labelist = labels.tolist()
                issuetest = []
                diff = []
                label = []

                idlist = issueid.tolist()
                p = []
                for i in range(0, len(difflist)):
                    diff_split = lsplit(difflist[i], numlist[i])
                    diff.extend(diff_split)
                    len1 = len(diff_split)
                    label.extend(lremove(labelist[i], 3)[:len1])
                    for numx in range(0, len1):
                        issuetest.append(desclist[i])
                idlist = issueid.tolist()
                for j in range(0, len(idlist)):
                    if target[j]:
                        p.append(allindex_cl(issueid, idlist[j]))
                issuetest = torch.LongTensor(issuetest)
                diff = torch.LongTensor(diff)
                label = torch.LongTensor(label)
                p = torch.LongTensor(p)
                message_inputs, code_inputs, desc_inputs, target, issuetest, diff, label, p = message_inputs.cuda(), code_inputs.cuda(), desc_inputs.cuda(), target.cuda(), issuetest.cuda(), diff.cuda(), label.cuda(), p.cuda()
                loss, c_l, n_l, sim_score = trans_model(message_inputs, code_inputs, desc_inputs, target.long(),
                                                        issuetest, diff,
                                                        label, p)
                loss = sum([lambda_weight[i, epoch] * loss[i] for i in range(3)])


                for n, p, prd, lb in zip(issueid.tolist(), list(commitid), sim_score, target.tolist()):
                    res.append((n, p, prd, lb))

                epoch_loss += loss.item()
        df = results_to_df(res)
        pd.DataFrame(df)
        df.reset_index(inplace=True)
        mrr = MRR(df)
        print("\t val MRR %f", '{:.4f}'.format(mrr))
        valid_avg_loss = epoch_loss / len(valid_data_loader)
        perplexity = math.exp(valid_avg_loss)
        perplexity = torch.tensor(perplexity).item()
        print('\t eval_loss: ', '{:.4f}'.format(valid_avg_loss))
        valid_ls.append(valid_avg_loss)
        print('\tperplexity: ', '{:.4f}'.format(perplexity))
        torch.save(trans_model.state_dict(), opt.output_model + '/'+str(epoch)+'_net.pt')
