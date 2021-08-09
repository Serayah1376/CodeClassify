# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 16:27:18 2021

@author: 10983
"""
import pandas as pd
import torch.utils.data as torch_data


class CodeDataset(torch_data.Dataset):
    def __init__(self, is_train_set=True):
        #定义label和0-103对应的字典（训练集中一共有104种标签）
        train_file = 'C:/Users/10983/py入门/GRUClassifier/train.pb'
        self.train_reader = pd.read_pickle(train_file)
        self.train_label = self.train_reader.loc[:, 'label']
        self.train_label_list = sorted(self.train_label.unique())
        
        filename = 'C:/Users/10983/py入门/GRUClassifier/train.pb' if is_train_set else 'C:/Users/10983/py入门/GRUClassifier/valid.pb'
        reader = pd.read_pickle(filename)
        self.codes_tmp = reader.loc[:, 'code'].values # 将代码列表放入codes中
        self.codes=self.cleanData(self.codes_tmp)
        self.len = len(self.codes)  # 记录样本的长度
        self.labels = reader.loc[:, 'label'].values  # 标签列表
        self.label_list = sorted(reader.loc[:, 'label'].unique())  # 去重排序后的标签列表
        self.label_dict = self.getLabelDict()
        self.label_num = len(self.label_list)  # 不同标签的个数  最终分类的总类别数
        

    # 根据索引获取code和对应的label
    def __getitem__(self, index):
        return self.codes[index], self.label_dict[self.labels[index]]

    # 返回数据集的长度
    def __len__(self):
        return self.len

    # 根据索引获取标签值
    def idx2label(self, index):
        return self.label_list[index]

    # 获取标签种类数
    def getLabelNum(self):
        return self.label_num

    # 获得标签对应的分类序号
    def getLabelDict(self):
        label_dict = dict()
        for idx, Label in enumerate(self.train_label_list, 0):
            label_dict[Label] = idx
        return label_dict
    
    #将'\n' '\t'以及无用空格全部删去
    def cleanData(self,code1):
        r=''
        i=0
        codes=[]
        while i<len(code1):
            s=code1[i].split("\n")
            r=''
            for ss in s:
                r+=ss.strip()
            codes.append(r)
            i+=1
        return codes
    


     
   











