'''
Named-Entity Recognition to Process Resumes
'''
import pandas as pd
import tensorflow as tf
import json
import random
import logging
import re
'''
1. Dataset Cleaning
'''
df_data = pd.read_json('ner.json', lines=True) # 由于JSON中每条数据是由空格分开的，所以需要使用lines=True（默认为False）
# 其余情况下调整orient参数可以调整读取后的方式

df_data = df_data.drop(['extras'], axis=1)
df_data['content'] = df_data['content'].str.replace('\n', ' ')  # 针对pd.series字符串来进行字符串操作


def mergeIntervals(intervals):
    sorted_by_lower_bound = sorted(intervals, key=lambda tup:tup[0])  # tup元素的第一个子元素进行排序
    merged = []
    for higher in sorted_by_lower_bound:
        if not merged:
            merged.append(higher)
        else:
            lower = merged[-1]
            if higher[0] <= lower[1]:
                if lower[2] is higher[2]:
                    upper_bound = max(lower[1], higher[1])
                    merged[-1] = (lower[0], upper_bound, lower[2])
                else:
                    if lower[1] > higher[1]:
                        merged[-1] = lower
                    else:
                        merged[-1] = (lower[0], higher[1], higher[2])
            else:
                merged.append(higher)
    return merged 



def get_entities(df):
    '''
    返回：e.g.：[(939, 956+1, 'skills'), ...]
    '''
    entities=[]

    for i in range(len(df)):
        entity=[]

        for annot in df['annotation'][i]:
            try:
                ent = annot['label'][0]
                start = annot['points'][0]['start']
                end = annot['points'][0]['end'] + 1
                entity.append((start, end, ent))
            except:
                pass
        
        entity = mergeIntervals(entity)
        entities.append(entity)
    return entities


df_data['entities'] = get_entities(df_data)
df_data.head()