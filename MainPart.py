# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 18:28:45 2021

@author: 10983
"""

import torch
import time
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader

torch.cuda.manual_seed_all(123)#设置随机数种子，使每一次初始化数值都相同

from Preparing_Data import CodeDataset
import RNNClassifier as RNN
import train

#parameters
HIDDEN_SIZE=256
BATCH_SIZE=16 #改为8试一下
N_LAYER=2  
N_EPOCHS=100
USE_GPU=True #使用GPU
gamma=0.1
max_len=229   #代码最大长度（95%分词后的代码低于此长度）

trainset=CodeDataset(is_train_set=True)
validset =CodeDataset(is_train_set=False)

#将分词后的codes进行padding：取每个batch的列表中的最大长度作为填充目标长度
def make_tensors(batch):
    codes=[item[0] for item in batch]
    labels=[item[1] for item in batch]
    list2=[]
    for s in codes:
        feature=[]
        #list1=[trainset.word2index[word] if trainset.word2index.get(word)!=None else 0 for word in s]
        for word in s:
            if word in trainset.word2index:
                feature.append(trainset.word2index[word])
            else:
                feature.append(trainset.word2index["<unk>"])
            #限制句子的最大长度，超出部分直接截去（这里选择截去结尾部分）
            if len(feature)==max_len:
                break
            
        #padding:填充1使长度相等
        need_add=max_len-len(feature)
        feature=feature + [trainset.word2index["<pad>"]] * need_add
        list2.append(feature)
    #将代码向量转化为张量
    seq_tensor=torch.LongTensor(list2)
    labels=torch.LongTensor(labels)     
    return trainset.create_tensor(seq_tensor),\
          trainset.create_tensor(labels)
  
#N_WORDS_test=testset.dicnum   

 
'''
N_CHARS：整个字母表元素个数
HIDDEN_SIZE：隐层维度
N_LABEL：有多少个分类
N_LAYER：用几层
'''
#数据的准备
#训练集
train_loader=DataLoader(trainset,batch_size=BATCH_SIZE,shuffle=False,collate_fn=make_tensors)
#验证集
valid_loader=DataLoader(validset,batch_size=BATCH_SIZE,shuffle=False,collate_fn=make_tensors)

#得到标签的种类数，决定模型最终输出维度的大小
#训练集中的类别是最全的，故使用训练集的label_num
N_LABEL=trainset.getLabelNum() 
#词典中词的个数，在使用词嵌入Embedding的时候传入的参数
N_WORDS_train=trainset.dicnum


classifier=RNN.RNNClassifier(N_WORDS_train,HIDDEN_SIZE,N_LABEL,n_layers=N_LAYER)
classifier.build()
#是否用GPU
if USE_GPU:
    device=torch.device('cuda:0')
    classifier.to(device)
    
#交叉熵损失
#criterion=torch.nn.CrossEntropyLoss()
#优化器
#optimizer=torch.optim.Adam(classifier.parameters(),lr=0.1)
#scheduler=torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma, last_epoch=-1)
#看训练的时间有多长
start=time.time()
print('Training for %d epochs...' % N_EPOCHS)

#存放训练的结果
acc_list=[]
mini_loss=100
acc_list=train.train(classifier,mini_loss,train_loader,valid_loader,start,trainset,validset)

#绘图部分看准确率的变化
epoch=np.arange(1,len(acc_list)+1,1)
acc_list=np.array(acc_list)
plt.plot(epoch,acc_list)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid()
plt.show()


