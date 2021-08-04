# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 16:27:18 2021

@author: 10983
"""
import pandas as pd

import torch.utils.data as torch_data

class CodeDataset(torch_data.Dataset):
    def __init__(self,is_train_set=True):
        filename='C:/Users/10983/py入门/GRUClassifier/train.pb' if is_train_set else 'C:/Users/10983/py入门/GRUClassifier/valid.pb'
        reader=pd.read_pickle(filename)
        self.codes=reader.loc[:,'code'].values #将代码列表放入codes中
        self.len=len(self.codes)  #记录样本的长度
        self.labels=reader.loc[:,'label'].values #标签列表
        self.label_list=sorted(reader.loc[:,'label'].unique()) #去重排序后的标签列表
        self.label_num=len(self.label_list)   #不同标签的个数  最终分类的总类别数
    
    #根据索引获取code和对应的label
    def __getitem__(self,index):
        return self.codes[index],self.labels[index]
    
    #返回数据集的长度
    def __len__(self):
        return self.len
    
    #根据索引获取标签值
    def idx2label(self,index):
        return self.label_list[index]
    
    #获取标签种类数
    def getLabelNum(self):
        return self.label_num
        