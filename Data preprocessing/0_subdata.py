# -*- coding: utf-8 -*-
# 分开运行 处理分词
import preprocessor
import csv
from tree_sitter import Language, Parser
import re

import pandas as pd

csv.field_size_limit(500 * 1024 * 1024)
lang = 'java'

LANGUAGE = Language('parser_lang\\build\\my-languages.so', lang)
parser = Parser()
parser.set_language(LANGUAGE)
newlist = []
dummy_link = pd.read_csv('../data/rawdata/ambari_link.csv', engine='python')
for index, row in dummy_link.iterrows():
    labelist = []
    Diff_processed = []
    difflist = eval(row['Diff'])
    tg = row['label']
    num = len(difflist)
    if tg == 0:
        labelist = [0] * num
    elif tg == 1:
        text = str(row['comment']) + str(row['summary']) + str(row['description'])
        text = text.lower()
        cf = eval(row['changed_files'])
        len1 = len(cf)
        if len1 == num:
            for i in range(0, len1):
                func_name = cf[i].split('.')[0].split('/')[-1].lower()
                if text.find(func_name) != -1:
                    labelist.append(1)
                else:
                    labelist.append(0)
        else:
            labelist = [1] * num
    for d in difflist:
        diff = re.sub(r'\<ROW.[0-9]*\>', "", str(d))
        diff = re.sub(r'\<CODE.[0-9]*\>', "", diff)
        diff = re.sub(r'@.*[0-9].*@', "", diff)
        try:
            dl = preprocessor.extract_codetoken(diff, parser, lang)
        except:
            print(dl)
        if len(dl) == 0:
            dl = preprocessor.processDiffCode(diff)
        Diff_processed.append(dl)
    list1 = [Diff_processed, labelist, num]
    newlist.append(list1)
    print(index)
pd.DataFrame(newlist, columns=['Diff_processed', 'labelist', 'num']).to_csv("../data/balancedata/java/isis_link.csv")
