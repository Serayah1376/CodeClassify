# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 18:28:45 2021

@author: 10983
"""

import torch
import time
import math
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader

import Preparing_Data as PD
import RNNClassifier as RNN
#parameters
HIDDEN_SIZE=100
BATCH_SIZE=128
N_LAYER=2
N_EPOCHS=100
N_CHARS=128 #使用ASCII表进行字符到数字的映射
USE_GPU=True #使用GPU

#数据的准备
trainset=PD.CodeDataset(is_train_set=True)
trainloader=DataLoader(trainset,batch_size=BATCH_SIZE,shuffle=True)
testset =PD.CodeDataset(is_train_set=False)
testloader=DataLoader(testset,batch_size=BATCH_SIZE,shuffle=False)

N_LABEL=trainset.getLabelNum() #得到标签的种类数，决定模型最终输出维度的大小
        
#计算运行的时间，转换分钟和秒的形式
def time_since(since):
    s=time.time()-since
    m=math.floor(s/60)
    s-=m*60
    return '%dm %ds'% (m,s)

#训练过程

def trainModel():
    total_loss=0
    for i,(codes,labels) in enumerate(trainloader,1):
        inputs,seq_lengths,target=classifier.make_tensors(codes,labels)
        output=classifier.forward(inputs,seq_lengths)
        loss=criterion(output,target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss+=loss.item()
        if i%10==0:
            print(f'[{time_since(start)}] Epoch {epoch}',end='')
            print(f'[{i * len(inputs)}/{len(trainset)}]',end='')
            print(f'loss={total_loss/(i*len(inputs))}')
        return total_loss

def testModel():
    correct=0
    total=len(testset)
    print('evaluating trained model...')
    with torch.no_grad():#测试不需要求梯度
        for i,(codes,labels) in enumerate(testloader,1):
            inputs,seq_lengths,target=classifier.make_tensors(codes,labels)
            output=classifier.forward(inputs,seq_lengths)
            pred=output.max(dim=1,keepdim=True)[1]
            correct+=pred.eq(target.view_as(pred)).sum().item()
        
        percent='%.2f' % (100 * correct /total)
        print(f'Test set: Accuracy {correct}/{total} {percent} %')
    return correct/total
   

if __name__=='__main__':
    #自己定义的模型
    '''
    N_CHARS：整个字母表元素个数
    HIDDEN_SIZE：隐层维度
    N_LABEL：有多少个分类
    N_LAYER：用几层的GPU
    '''
    classifier=RNN.RNNClassifier(N_CHARS,HIDDEN_SIZE,N_LABEL,N_LAYER)
    #是否用GPU
    if USE_GPU:
        device=torch.device('cuda:0')
        classifier.to(device)
    
    #交叉熵损失
    criterion=torch.nn.CrossEntropyLoss()
    #优化器
    optimizer=torch.optim.Adam(classifier.parameters(),lr=0.01)
    
    #看训练的时间有多长
    start=time.time()
    print('Training for %d epochs...' % N_EPOCHS)
    #存放训练的结果
    acc_list=[]
    for epoch in range(1,N_EPOCHS+1):
        trainModel()
        acc=testModel()
        acc_list.append(acc) 
    
    #绘图部分看准确率的变化
    epoch=np.arange(1,len(acc_list)+1,1)
    acc_list=np.array(acc_list)
    plt.plot(epoch,acc_list)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid()
    plt.show()
