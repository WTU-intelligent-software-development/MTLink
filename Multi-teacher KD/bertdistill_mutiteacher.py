# -*- coding: utf-8 -*-
# @Time : 2023/9/30 18:22
# @Author : dy
# coding=gb2312
import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from tqdm import tqdm
import numpy as np
import random
from transformers import (RobertaConfig, RobertaModel,
                        AdamW, BertConfig, BertModel, BertTokenizer,
                        RobertaTokenizer,get_linear_schedule_with_warmup, BertForMaskedLM,
                        get_cosine_with_hard_restarts_schedule_with_warmup)
from transformers.modeling_utils import PreTrainedModel
from pathlib import Path
import argparse
from torch.utils.data import DataLoader, SequentialSampler
from dataset import make_data, MyDataSet, MySampler
import textbrewer
from textbrewer import GeneralDistiller
from textbrewer import TrainingConfig, DistillationConfig
import logging
from transformers import AutoTokenizer, AutoModel
import pandas as pd
logger = logging.getLogger()
handler = logging.StreamHandler()
logger.addHandler(handler)
logger.setLevel(logging.INFO)

def set_seed(seed=45):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
set_seed()


MODEL_CLASSES = {
    'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer),
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_basic_model(config=None):
    config_class, model_class, tokenizer_class = MODEL_CLASSES['roberta']
    if not config:
        config = config_class.from_pretrained('microsoft/codebert-base')
    else: config = config
    config.output_hidden_states = True
    # print(config)
    # tokenizer = tokenizer_class.from_pretrained('microsoft/codebert-base')
    nl_model = model_class.from_pretrained('microsoft/codebert-base',
                                    config=config, from_tf=False)
    tokenizer = tokenizer_class.from_pretrained('microsoft/codebert-base', do_lower_case=True)
    nl_model.to(device)
    return nl_model, config, tokenizer

def get_basic_model_2(config=None):
    config_class, model_class, tokenizer_class = MODEL_CLASSES['roberta']
    if not config:
        config = config_class.from_pretrained('microsoft/graphcodebert-base')
    else: config = config
    config.output_hidden_states = True
    # print(config)
    # tokenizer = tokenizer_class.from_pretrained('microsoft/codebert-base')
    nl_model = model_class.from_pretrained('microsoft/graphcodebert-base',
                                    config=config, from_tf=False)
    tokenizer = tokenizer_class.from_pretrained('microsoft/graphcodebert-base', do_lower_case=True)
    nl_model.to(device)
    return nl_model, config, tokenizer
# teacher model,这里的config和tokenizer没有被用到
model_Ts = []

teacher_model, config, tokenizer = get_basic_model()
tokenizer = RobertaTokenizer.from_pretrained('microsoft/codebert-base')
model_Ts.append(teacher_model)

teacher_model, config, tokenizer = get_basic_model_2()
tokenizer = RobertaTokenizer.from_pretrained('microsoft/graphcodebert-base')
model_Ts.append(teacher_model)

# student model
config_path = Path('tiny_bert_config.json')
student_config = RobertaConfig.from_json_file(config_path)
student_model, config, tokenizer = get_basic_model(config=student_config)

print('loading data.................')
files = os.listdir('../data/')
corpus_df = pd.DataFrame()
def mergedf(corpus_df, path, files):
    c = 0
    for file in files:
        if file[-8:] == 'link.csv' :
            data = pd.read_csv(path+file,encoding='GBK')
            corpus_df = pd.concat([corpus_df, data])
    return corpus_df
df = mergedf(corpus_df, '../data/', files)
examples = make_data(df,tokenizer)
train_data = MyDataSet(examples)
my_sampler = MySampler(train_data, 128)
mydataloader = DataLoader(train_data, batch_sampler=my_sampler)
# for step, batch in tqdm(enumerate(mydataloader)):
#     print(batch)
#     exit()

num_epochs = 5
num_training_steps = len(mydataloader) / 16 *num_epochs
print(f'num of training steps:{num_training_steps}')
# Optimizer and learning rate scheduler
optimizer = AdamW(student_model.parameters(), lr=1e-4)

scheduler_class = get_linear_schedule_with_warmup
# arguments dict except 'optimizer'
scheduler_args = {'num_warmup_steps':int(0.1*num_training_steps), 'num_training_steps':num_training_steps}

def simple_adaptor(batch, model_outputs):
    teacher_1= model_outputs[0].hidden_states
    teacher_2 = model_outputs[1].hidden_states
    # 解压缩元组并计算平均值
    result=[]
    for subtuple1, subtuple2 in zip(teacher_1, teacher_2):
        result.append(sum(subtuple1, subtuple2)/2)

    #result = [[(a + b) / 2 for a, b in zip(subtuple1, subtuple2)] for subtuple1, subtuple2 in zip(teacher_1, teacher_2)]
    return {'hidden': tuple(result)}

#def simple_adaptor(batch, model_outputs):
#    return {'hidden': model_outputs[1].hidden_states}

def simple_adaptor2(batch, model_outputs):
    return {'hidden': model_outputs.hidden_states}


distill_config = DistillationConfig(
    intermediate_matches=[
     {'layer_T': 1, 'layer_S': 0, 'feature':'hidden', 'loss': 'hidden_mse','weight' : 1},
     {'layer_T': 4, 'layer_S': 1, 'feature':'hidden', 'loss': 'hidden_mse','weight' : 1},
    #  {'layer_T': 8, 'layer_S': 2, 'feature':'hidden', 'loss': 'hidden_mse','weight' : 1},
    #  {'layer_T': 11, 'layer_S': 3, 'feature':'hidden', 'loss': 'hidden_mse','weight' : 1},
    ])
train_config = TrainingConfig()

distiller = GeneralDistiller(
    train_config=train_config, distill_config=distill_config,
    model_T=model_Ts, model_S=student_model,
    adaptor_T=simple_adaptor, adaptor_S=simple_adaptor2)

with distiller:

    distiller.train(optimizer, scheduler_class=scheduler_class, scheduler_args=scheduler_args,
                    dataloader=mydataloader,
                    callback=None, max_grad_norm=1, num_epochs = num_epochs)
print('codebert')

