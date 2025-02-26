# -*- coding: utf-8 -*-
# @Time : 2023/10/4 21:03
# @Author : dy
from __future__ import print_function
from __future__ import absolute_import
import torch
import torch.nn as nn
from transformers import (RobertaConfig, RobertaModel,
                          AdamW, BertConfig, BertModel, BertTokenizer,
                          RobertaTokenizer, get_linear_schedule_with_warmup, BertForMaskedLM,
                          get_cosine_with_hard_restarts_schedule_with_warmup)
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from pathlib import Path


class AvgPooler(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_size = 768
        self.pooler = torch.nn.AdaptiveAvgPool2d((1, 768))

    def forward(self, hidden_states):
        return self.pooler(hidden_states).view(-1, self.hidden_size)


class RelationClassifyHeader(nn.Module):
    """
    H2:
    use averaging pooling across tokens to replace first_token_pooling
    """

    def __init__(self):
        super().__init__()
        self.hidden_size = 768
        self.code_pooler = AvgPooler()
        self.text_pooler = AvgPooler()

        self.dense = nn.Linear(768 * 3, 768)
        self.dropout = nn.Dropout(0.1)
        self.output_layer = nn.Linear(768, 2)

    def forward(self, code_hidden, text_hidden):
        pool_code_hidden = self.code_pooler(code_hidden)
        pool_text_hidden = self.text_pooler(text_hidden)
        diff_hidden = torch.abs(pool_code_hidden - pool_text_hidden)
        concated_hidden = torch.cat((pool_code_hidden, pool_text_hidden), 1)
        concated_hidden = torch.cat((concated_hidden, diff_hidden), 1)

        x = self.dropout(concated_hidden)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.output_layer(x)
        return x


class RelationClassifyHeader_2(nn.Module):
    """
    H2:
    use averaging pooling across tokens to replace first_token_pooling
    """

    def __init__(self):
        super().__init__()
        self.hidden_size = 768
        self.code_pooler = AvgPooler()
        self.text_pooler = AvgPooler()

    def forward(self, code_hidden, text_hidden):
        pool_code_hidden = self.code_pooler(code_hidden)
        pool_text_hidden = self.text_pooler(text_hidden)
        querys_size = pool_code_hidden.shape[1]
        code_size = pool_text_hidden.shape[1]
        fc_input_size = querys_size + code_size
        num_class = 2
        self.fc = nn.Linear(fc_input_size, int(fc_input_size / 2))
        self.fc.cuda()
        self.fc1 = nn.Linear(int(fc_input_size / 2),
                             int(fc_input_size / 4))
        self.fc1.cuda()
        self.fc2 = nn.Linear(int(fc_input_size / 4), num_class)
        self.fc2.cuda()
        self.relu = nn.ReLU()

        combine_output = torch.cat([pool_text_hidden, pool_code_hidden], dim=-1)
        logits = self.fc(combine_output)
        logits = self.fc1(logits)
        logits = self.fc2(logits)
        return logits


class Multi_Model(nn.Module):
    def __init__(self):
        super(Multi_Model, self).__init__()
        #config_path = Path('../Dstill/four_bert_config.json')
        config_path = Path('../Dstill/two_bert_config.json')

        student_config = RobertaConfig.from_json_file(config_path)
        codeBert = RobertaModel(student_config)
        codeBert.load_state_dict(torch.load('../Dstill/cal/saved_models_cal_teacher1_2layer/gs235.pkl'), strict=False)

        self.encoder = codeBert
        self.loss_fct = CrossEntropyLoss()
        self.infonce = InfoNce_2()
        self.cls = RelationClassifyHeader()
        self.cls_2 = RelationClassifyHeader_2()


    def getLoss(self, score, t):
        rs = t - score
        rs = torch.abs(rs)
        #rs = torch.lstsq(rs)
        return torch.sum(rs)

    def get_sim_score(self, text_hidden, code_hidden):
        logits = self.cls(text_hidden=text_hidden, code_hidden=code_hidden)
        sim_scores = torch.softmax(logits, 1).data.tolist()
        return sim_scores

    def forward(self, message_inputs, code_inputs, desc_inputs, target, querys_inputs=None, diff_inputs=None,
                label=None, p=None, mode='train'):  # 变动
        t = 0.07
        loss=[]
        # data embedding
        commit_inputs = torch.cat([message_inputs, code_inputs], dim=1)
        message = self.encoder(commit_inputs, attention_mask=commit_inputs.ne(1))[0]
        desc = self.encoder(desc_inputs, attention_mask=desc_inputs.ne(1))[0]

        # task1
        commit = self.cls.code_pooler(message)
        issue = self.cls.text_pooler(desc)
        sim = F.cosine_similarity(commit, issue)
        rel_loss = self.getLoss(sim, target)  # 目标函数

        # contrastive learning
        commit_vec = F.normalize(commit, p=2, dim=-1, eps=1e-5)
        issue_vec = F.normalize(issue, p=2, dim=-1, eps=1e-5)
        link_repr = torch.cat((issue_vec, commit_vec), 1)  # [batchsize,2*d_model]
        sims_matrix = torch.matmul(link_repr, link_repr.t())  # [batchsize,batchsize]
        sims_matrix_1 = sims_matrix[target == 1]
        sims_matrix_0 = sims_matrix[target == 0]
        cl_loss = self.infonce(sims_matrix_1, sims_matrix_0, p, t)  # 对抗学习

        if mode == 'test':
            return rel_loss, sim.data.tolist()

        # task2
        querys = self.encoder(querys_inputs, attention_mask=querys_inputs.ne(1))[0]
        codes = self.encoder(diff_inputs, attention_mask=diff_inputs.ne(1))[0]
        # logits = self.cls(codes, querys)
        logits = self.cls_2(codes, querys)

        sub_loss = self.loss_fct(logits.view(-1, 2), label.view(-1))  # 交叉熵函数



        loss.append(sub_loss)
        loss.append(cl_loss)
        loss.append(rel_loss)
        #loss = rel_loss + cl_loss + sub_loss
        return loss, message, desc, sim.data.tolist()


# infonceloss
class InfoNce(nn.Module):
    def __init__(self):
        super(InfoNce, self).__init__()

    def forward(self, sims, labels, nce):
        f = lambda x: torch.exp(x / nce)
        new_sim = f(sims)
        return -torch.log(new_sim[labels == 1] / new_sim.sum(1)).mean()  # 真链接/正样本


class InfoNce_2(nn.Module):
    def __init__(self):
        super(InfoNce_2, self).__init__()

    def forward(self, sims_1, sims_0, labels, nce):
        f = lambda x: torch.exp(x / nce)
        new_sim_1 = f(sims_1)
        new_sim_0 = f(sims_0)
        sum_1 = new_sim_1[labels == 1]
        for i in new_sim_1[labels == 0]:
            sum_1 = sum_1 + i
        return -torch.log(new_sim_1[labels == 1] / (sum_1)).mean()
